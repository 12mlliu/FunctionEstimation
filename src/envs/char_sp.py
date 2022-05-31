# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import io
import re
import sys
import math
import itertools
from collections import OrderedDict
import numpy as np
np.seterr(invalid='ignore')#suppress RuntimeWarning for nan value
import numexpr as ne
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.cache import clear_cache
from sympy.integrals.risch import NonElementaryIntegral
from sympy.calculus.util import AccumBounds

from ..utils import bool_flag
from ..utils import timeout, TimeoutError
from .sympy_utils import remove_root_constant_terms, reduce_coefficients, reindex_coefficients
from .sympy_utils import extract_non_constant_subtree, simplify_const_with_coeff, simplify_equa_diff, clean_degree2_solution
from .sympy_utils import remove_mul_const, has_inf_nan, has_I, simplify


'''
import argparse
def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")
'''
CLEAR_SYMPY_CACHE_FREQ = 10000


SPECIAL_WORDS = ['<s>', '</s>', '<pad>', '(', ')']
#SPECIAL_WORDS = SPECIAL_WORDS + [f'<SPECIAL_{i}>' for i in range(len(SPECIAL_WORDS), 10)]


INTEGRAL_FUNC = {sp.erf, sp.erfc, sp.erfi, sp.erfinv, sp.erfcinv, sp.expint, sp.Ei, sp.li, sp.Li, sp.Si, sp.Ci, sp.Shi, sp.Chi, sp.fresnelc, sp.fresnels}
EXP_OPERATORS = {'exp', 'sinh', 'cosh'}
EVAL_SYMBOLS = {'x', 'y', 'z', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9'}
EVAL_VALUES = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 2.1, 3.1]
EVAL_VALUES = EVAL_VALUES + [-x for x in EVAL_VALUES]

TEST_ZERO_VALUES = [0.1, 0.9, 1.1, 1.9]
TEST_ZERO_VALUES = [-x for x in TEST_ZERO_VALUES] + TEST_ZERO_VALUES
ZERO_THRESHOLD = 1e-13


logger = getLogger()


class ValueErrorExpression(Exception):
    pass


class UnknownSymPyOperator(Exception):
    pass


class InvalidPrefixExpression(Exception):

    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)

def big_value_filter(data_y):
    #this functions like np.isnan()
    #input: data_y is the generated dataset
    #output: pos False is for valid value, True is for invalid value
    pos = np.isnan(data_y) #detect nan
    #pos=[0 if e else 1 for e in pos] # 0 is for invalid data
    for i,v in enumerate(data_y):
        #detect nan
        if (v >=1e10) or (v<=1e-10 and v>=-1e-10) or (v<=-1e10):
            pos[i]=True
    return pos
    
def count_nested_exp(s):
    """
    Return the maximum number of nested exponential functions in an infix expression.
    """
    stack = []
    count = 0
    max_count = 0
    for v in re.findall('[+-/*//()]|[a-zA-Z0-9]+', s):
        if v == '(':
            stack.append(v)
        elif v == ')':
            while True:
                x = stack.pop()
                if x in EXP_OPERATORS:
                    count -= 1
                if x == '(':
                    break
        else:
            stack.append(v)
            if v in EXP_OPERATORS:
                count += 1
                max_count = max(max_count, count)
    assert len(stack) == 0
    return max_count


def is_valid_expr(s):
    """
    Check that we are able to evaluate an expression (and that it will not blow in SymPy evaluation).
    """
    s = s.replace('Derivative(f(x),x)', '1')
    s = s.replace('Derivative(1,x)', '1')
    s = s.replace('(E)', '(exp(1))')
    s = s.replace('(I)', '(1)')
    s = s.replace('(pi)', '(1)')
    s = re.sub(r'(?<![a-z])(f|g|h|Abs|sign|ln|sin|cos|tan|sec|csc|cot|asin|acos|atan|asec|acsc|acot|tanh|sech|csch|coth|asinh|acosh|atanh|asech|acoth|acsch)\(', '(', s)
    count = count_nested_exp(s)
    if count >= 4:
        return False
    for v in EVAL_VALUES:
        try:
            local_dict = {s: (v + 1e-4 * i) for i, s in enumerate(EVAL_SYMBOLS)}
            value = ne.evaluate(s, local_dict=local_dict).item()
            if not (math.isnan(value) or math.isinf(value)):
                return True
        except (FloatingPointError, ZeroDivisionError, TypeError, MemoryError):
            continue
    return False


def eval_test_zero(eq):
    """
    Evaluate an equation by replacing all its free symbols with random values.
    """
    variables = eq.free_symbols
    assert len(variables) <= 3
    outputs = []
    for values in itertools.product(*[TEST_ZERO_VALUES for _ in range(len(variables))]):
        _eq = eq.subs(zip(variables, values)).doit()
        outputs.append(float(sp.Abs(_eq.evalf())))
    return outputs


class CharSPEnvironment(object):


    # https://docs.sympy.org/latest/modules/functions/elementary.html#real-root

    SYMPY_OPERATORS = {
        # Elementary functions
        sp.Add: 'add',
        sp.Mul: 'mul',
        sp.Pow: 'pow',
        sp.exp: 'exp',
        sp.log: 'ln',
        sp.Abs: 'abs',
        sp.sign: 'sign',
        # Trigonometric Functions
        sp.sin: 'sin',
        sp.cos: 'cos',
        sp.tan: 'tan',
        sp.cot: 'cot',
        sp.sec: 'sec',
        sp.csc: 'csc',
        # Trigonometric Inverses
        sp.asin: 'asin',
        sp.acos: 'acos',
        sp.atan: 'atan',
        sp.acot: 'acot',
        sp.asec: 'asec',
        sp.acsc: 'acsc',
        # Hyperbolic Functions
        sp.sinh: 'sinh',
        sp.cosh: 'cosh',
        sp.tanh: 'tanh',
        sp.coth: 'coth',
        sp.sech: 'sech',
        sp.csch: 'csch',
        # Hyperbolic Inverses
        sp.asinh: 'asinh',
        sp.acosh: 'acosh',
        sp.atanh: 'atanh',
        sp.acoth: 'acoth',
        sp.asech: 'asech',
        sp.acsch: 'acsch',
    }

    OPERATORS = {
        # Elementary functions
        'add': 2,
        'sub': 2,
        'mul': 2,
        'div': 2,
        'pow': 2,
        'rac': 2,
        'inv': 1,
        'pow2': 1,
        'pow3': 1,
        'pow4': 1,
        'pow5': 1,
        'sqrt': 1,
        'exp': 1,
        'ln': 1,
        'abs': 1,
        'sign': 1,
        # Trigonometric Functions
        'sin': 1,
        'cos': 1,
        'tan': 1,
        'cot': 1,
        'sec': 1,
        'csc': 1,
        # Trigonometric Inverses
        'asin': 1,
        'acos': 1,
        'atan': 1,
        'acot': 1,
        'asec': 1,
        'acsc': 1,
        # Hyperbolic Functions
        'sinh': 1,
        'cosh': 1,
        'tanh': 1,
        'coth': 1,
        'sech': 1,
        'csch': 1,
        # Hyperbolic Inverses
        'asinh': 1,
        'acosh': 1,
        'atanh': 1,
        'acoth': 1,
        'asech': 1,
        'acsch': 1,
    }

    def __init__(self, params):
        
        self.datalength = params.datalength
        self.max_int = params.max_int #5 max value of sampled integers
        self.max_ops = params.max_ops #5 maximum number of operators at generation
        #self.max_ops_G = params.max_ops_G
        self.int_base = params.int_base #10
        self.balanced = params.balanced #false
        self.positive = params.positive #true
        #self.precision = params.precision 
        self.n_variables = params.n_variables # 1 only variables x right now
        #self.n_coefficients = params.n_coefficients # no coefficients right now
        self.max_len = params.max_len #64 maximum length of generated equations,equal to T or block_size
        #self.clean_prefix_expr = params.clean_prefix_expr
        assert self.max_int >= 1
        assert abs(self.int_base) >= 2
        #assert self.precision >= 2

        # parse operators with their weights
        self.operators = sorted(list(self.OPERATORS.keys()))
        #operators and thier weight
        #params.operators  = "add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1"
        ops = params.operators.split(',')
        ops = sorted([x.split(':') for x in ops])
        assert len(ops) >= 1 and all(o in self.OPERATORS for o, _ in ops)
        self.all_ops = [o for o, _ in ops]
        self.una_ops = [o for o, _ in ops if self.OPERATORS[o] == 1]
        self.bin_ops = [o for o, _ in ops if self.OPERATORS[o] == 2]
        logger.info(f"Unary operators: {self.una_ops}")
        logger.info(f"Binary operators: {self.bin_ops}")
        self.all_ops_probs = np.array([float(w) for _, w in ops]).astype(np.float64)
        self.una_ops_probs = np.array([float(w) for o, w in ops if self.OPERATORS[o] == 1]).astype(np.float64)
        self.bin_ops_probs = np.array([float(w) for o, w in ops if self.OPERATORS[o] == 2]).astype(np.float64)
        self.all_ops_probs = self.all_ops_probs / self.all_ops_probs.sum()
        self.una_ops_probs = self.una_ops_probs / self.una_ops_probs.sum()
        self.bin_ops_probs = self.bin_ops_probs / self.bin_ops_probs.sum()

        assert len(self.all_ops) == len(set(self.all_ops)) >= 1
        assert set(self.all_ops).issubset(set(self.operators))
        assert len(self.all_ops) == len(self.una_ops) + len(self.bin_ops)

        # symbols / elements
        self.constants = ['pi', 'E']
        #self.constants = ['pi']
        self.variables = OrderedDict({
            'x': sp.Symbol('x', real=True, nonzero=True),  # , positive=True
        #    'y': sp.Symbol('y', real=True, nonzero=True),  # , positive=True
        #    'z': sp.Symbol('z', real=True, nonzero=True),  # , positive=True
        #    't': sp.Symbol('t', real=True, nonzero=True),  # , positive=True
        })
        #self.coefficients = OrderedDict({
        #    f'a{i}': sp.Symbol(f'a{i}', real=True)
        #    for i in range(10)
        #})
        #self.functions = OrderedDict({
        #    'f': sp.Function('f', real=True, nonzero=True),
        #    'g': sp.Function('g', real=True, nonzero=True),
        #    'h': sp.Function('h', real=True, nonzero=True),
        #})
        #self.symbols = ['I', 'INT+', 'INT-', 'INT', 'FLOAT', '-', '.', '10^', 'Y', "Y'", "Y''"]
        self.symbols = [ 'INT+', 'INT-', 'INT']
        #if self.balanced:
        #    assert self.int_base > 2
        #    max_digit = (self.int_base + 1) // 2
        #    self.elements = [str(i) for i in range(max_digit - abs(self.int_base), max_digit)]
        #else:
        #get elements from self.int_base
        self.elements = [str(i) for i in range(abs(self.int_base))]
        assert 1 <= self.n_variables <= len(self.variables)
        #assert 0 <= self.n_coefficients <= len(self.coefficients)
        #assert all(k in self.OPERATORS for k in self.functions.keys())
        assert all(v in self.OPERATORS for v in self.SYMPY_OPERATORS.values())

        # SymPy elements
        self.local_dict = {}
        for k, v in list(self.variables.items()):# + list(self.coefficients.items()) + list(self.functions.items()):
            assert k not in self.local_dict
            self.local_dict[k] = v

        # vocabulary
        self.words = SPECIAL_WORDS + self.constants + list(self.variables.keys()) + self.operators + self.symbols + self.elements
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)
        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        logger.info(f"words: {self.word2id}")

     
        #leaf probability [0.5,0.25,0.25] # variables integers and constants
        s = [float(x) for x in params.leaf_probs.split(',')]
        assert len(s) == 3 and all(x >= 0 for x in s)
        self.leaf_probs = np.array(s).astype(np.float64)
        self.leaf_probs = self.leaf_probs / self.leaf_probs.sum()
        assert self.leaf_probs[0] > 0
        #assert (self.leaf_probs[1] == 0) == (self.n_coefficients == 0)

        # possible leaves
        self.n_leaves = self.n_variables # + self.n_coefficients
        if self.leaf_probs[1] > 0:
            self.n_leaves += self.max_int * (1 if self.positive else 2)
        if self.leaf_probs[2] > 0:
            self.n_leaves += len(self.constants)
        logger.info(f"{self.n_leaves} possible leaves.")

        # generation parameters
        self.nl = 1  # self.n_leaves
        self.p1 = 1  # len(self.una_ops)
        self.p2 = 1  # len(self.bin_ops)

        # initialize distribution for binary and unary-binary trees
        self.bin_dist = self.generate_bin_dist(params.max_ops)
        self.ubi_dist = self.generate_ubi_dist(params.max_ops)

        # rewrite expressions
        self.rewrite_functions = [x for x in params.rewrite_functions.split(',') if x != '']
        assert len(self.rewrite_functions) == len(set(self.rewrite_functions))
        assert all(x in ['expand', 'factor', 'expand_log', 'logcombine', 'powsimp', 'simplify'] for x in self.rewrite_functions)

        # valid check
        logger.info(f"Checking expressions in {str(EVAL_VALUES)}")

    def batch_sequences(self, sequences):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        <eos_index> sin x <eos_index>  <pad_index>
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sequences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.pad_index)
        assert lengths.min().item() > 2

        sent[0] = self.eos_index
        for i, s in enumerate(sequences):
            sent[1:lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.eos_index
            
        return sent, lengths
            
    def batch_sequences_data(self, sequences):
        """
        the process of generating dataset can make sure that they have the same length
        therefore, there is no need to pad
        but we need transpose
        """
        lengths = torch.LongTensor([len(s) for s in sequences]) #n_embd*T
        #assert all length are equal
        assert all(x == lengths[0] for x in lengths), "not all dataset has the same length"
        sent = sequences.transpose(0,1) #(slen,batch)

        return sent, lengths

    def generate_bin_dist(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(0, n) = 0
            D(1, n) = C_n (n-th Catalan number)
            D(e, n) = D(e - 1, n + 1) - D(e - 2, n + 1)
        """
        # initialize Catalan numbers
        catalans = [1]
        for i in range(1, 2 * max_ops + 1):
            catalans.append((4 * i - 2) * catalans[i - 1] // (i + 1))

        # enumerate possible trees
        D = []
        for e in range(max_ops + 1):  # number of empty nodes
            s = []
            for n in range(2 * max_ops - e + 1):  # number of operators
                if e == 0:
                    s.append(0)
                elif e == 1:
                    s.append(catalans[n])
                else:
                    s.append(D[e - 1][n + 1] - D[e - 2][n + 1])
            D.append(s)
        return D

    def generate_ubi_dist(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(0, n) = 0
            D(e, 0) = L ** e
            D(e, n) = L * D(e - 1, n) + p_1 * D(e, n - 1) + p_2 * D(e + 1, n - 1)
        """
        # enumerate possible trees
        # first generate the tranposed version of D, then transpose it
        D = []
        D.append([0] + ([self.nl ** i for i in range(1, 2 * max_ops + 1)]))
        for n in range(1, 2 * max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * max_ops - n + 1):  # number of empty nodes
                s.append(self.nl * s[e - 1] + self.p1 * D[n - 1][e] + self.p2 * D[n - 1][e + 1])
            D.append(s)
        assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
        D = [[D[j][i] for j in range(len(D)) if i < len(D[j])] for i in range(max(len(x) for x in D))]
        return D

    def write_int(self, val):
        """
        Convert a decimal integer to a representation in the given base.
        The base can be negative.
        In balanced bases (positive), digits range from -(base-1)//2 to (base-1)//2
        """
        base = self.int_base
        balanced = self.balanced
        res = []
        max_digit = abs(base)
        if balanced:
            max_digit = (base - 1) // 2
        else:
            if base > 0:
                neg = val < 0
                val = -val if neg else val
        while True:
            rem = val % base
            val = val // base
            if rem < 0 or rem > max_digit:
                rem -= base
                val += 1
            res.append(str(rem))
            if val == 0:
                break
        if base < 0 or balanced:
            res.append('INT')
        else:
            res.append('INT-' if neg else 'INT+')
        return res[::-1]

    def parse_int(self, lst):
        """
        Parse a list that starts with an integer.
        Return the integer value, and the position it ends in the list.
        """
        base = self.int_base
        balanced = self.balanced
        val = 0
        if not (balanced and lst[0] == 'INT' or base >= 2 and lst[0] in ['INT+', 'INT-'] or base <= -2 and lst[0] == 'INT'):
            raise InvalidPrefixExpression(f"Invalid integer in prefix expression")
        i = 0
        for x in lst[1:]:
            if not (x.isdigit() or x[0] == '-' and x[1:].isdigit()):
                break
            val = val * base + int(x)
            i += 1
        if base > 0 and lst[0] == 'INT-':
            val = -val
        return val, i + 1

    def sample_next_pos_ubi(self, nb_empty, nb_ops, rng):
        """
        Sample the position of the next node (unary-binary case).
        Sample a position in {0, ..., `nb_empty` - 1}, along with an arity.
        """
        assert nb_empty > 0
        assert nb_ops > 0
        probs = []
        for i in range(nb_empty):
            probs.append((self.nl ** i) * self.p1 * self.ubi_dist[nb_empty - i][nb_ops - 1])
        for i in range(nb_empty):
            probs.append((self.nl ** i) * self.p2 * self.ubi_dist[nb_empty - i + 1][nb_ops - 1])
        probs = [p / self.ubi_dist[nb_empty][nb_ops] for p in probs]
        probs = np.array(probs, dtype=np.float64)
        e = rng.choice(2 * nb_empty, p=probs)
        arity = 1 if e < nb_empty else 2
        e = e % nb_empty
        return e, arity

    def get_leaf(self, max_int, rng):
        """
        Generate a leaf.
        """
        self.leaf_probs
        leaf_type = rng.choice(3, p=self.leaf_probs)
        if leaf_type == 0:
            return [list(self.variables.keys())[rng.randint(self.n_variables)]]
        #elif leaf_type == 1:
        #    return [list(self.coefficients.keys())[rng.randint(self.n_coefficients)]]
        elif leaf_type == 1:
            c = rng.randint(1, max_int )
            c = c if (self.positive or rng.randint(2) == 0) else -c
            return self.write_int(c)
        else:
            return [self.constants[rng.randint(len(self.constants))]]

    def _generate_expr(self, nb_total_ops, max_int, rng, require_x=False, require_y=False, require_z=False):
        """
        Create a tree with exactly `nb_total_ops` operators.
        """
        stack = [None]
        nb_empty = 1  # number of empty nodes
        l_leaves = 0  # left leaves - None states reserved for leaves
        t_leaves = 1  # total number of leaves (just used for sanity check)

        # create tree
        for nb_ops in range(nb_total_ops, 0, -1):

            # next operator, arity and position
            skipped, arity = self.sample_next_pos_ubi(nb_empty, nb_ops, rng)
            if arity == 1:
                op = rng.choice(self.una_ops, p=self.una_ops_probs)
            else:
                op = rng.choice(self.bin_ops, p=self.bin_ops_probs)

            nb_empty += self.OPERATORS[op] - 1 - skipped  # created empty nodes - skipped future leaves
            t_leaves += self.OPERATORS[op] - 1            # update number of total leaves
            l_leaves += skipped                           # update number of left leaves

            # update tree
            pos = [i for i, v in enumerate(stack) if v is None][l_leaves]
            stack = stack[:pos] + [op] + [None for _ in range(self.OPERATORS[op])] + stack[pos + 1:]

        # sanity check
        assert len([1 for v in stack if v in self.all_ops]) == nb_total_ops
        assert len([1 for v in stack if v is None]) == t_leaves

        # create leaves
        # optionally add variables x, y, z if possible
        assert not require_z or require_y
        assert not require_y or require_x
        leaves = [self.get_leaf(max_int, rng) for _ in range(t_leaves)]
        if require_z and t_leaves >= 2:
            leaves[1] = ['z']
        if require_y:
            leaves[0] = ['y']
        if require_x and not any(len(leaf) == 1 and leaf[0] == 'x' for leaf in leaves):
            leaves[-1] = ['x']
        rng.shuffle(leaves)

        # insert leaves into tree
        for pos in range(len(stack) - 1, -1, -1):
            if stack[pos] is None:
                stack = stack[:pos] + leaves.pop() + stack[pos + 1:]
        assert len(leaves) == 0

        return stack

    def write_infix(self, token, args):
        """
        Infix representation.
        Convert prefix expressions to a format that SymPy can parse.
        """
        if token == 'add':
            return f'({args[0]})+({args[1]})'
        elif token == 'sub':
            return f'({args[0]})-({args[1]})'
        elif token == 'mul':
            return f'({args[0]})*({args[1]})'
        elif token == 'div':
            return f'({args[0]})/({args[1]})'
        elif token == 'pow':
            return f'({args[0]})**({args[1]})'
        elif token == 'rac':
            return f'({args[0]})**(1/({args[1]}))'
        elif token == 'abs':
            return f'Abs({args[0]})'
        elif token == 'inv':
            return f'1/({args[0]})'
        elif token == 'pow2':
            return f'({args[0]})**2'
        elif token == 'pow3':
            return f'({args[0]})**3'
        elif token == 'pow4':
            return f'({args[0]})**4'
        elif token == 'pow5':
            return f'({args[0]})**5'
        elif token in ['sign', 'sqrt', 'exp', 'ln', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'asin', 'acos', 'atan', 'acot', 'asec', 'acsc', 'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch', 'asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch']:
            return f'{token}({args[0]})'
        elif token == 'derivative':
            return f'Derivative({args[0]},{args[1]})'
        elif token == 'f':
            return f'f({args[0]})'
        elif token == 'g':
            return f'g({args[0]},{args[1]})'
        elif token == 'h':
            return f'h({args[0]},{args[1]},{args[2]})'
        elif token.startswith('INT'):
            return f'{token[-1]}{args[0]}'
        else:
            return token
        raise InvalidPrefixExpression(f"Unknown token in prefix expression: {token}, with arguments {args}")

    def _prefix_to_infix(self, expr):
        """
        Parse an expression in prefix mode, and output it in either:
          - infix mode (returns human readable string)
          - develop mode (returns a dictionary with the simplified expression)
        """
        if len(expr) == 0:
            raise InvalidPrefixExpression("Empty prefix list.")
        t = expr[0]
        if t in self.operators:
            args = []
            l1 = expr[1:]
            for _ in range(self.OPERATORS[t]):
                i1, l1 = self._prefix_to_infix(l1)
                args.append(i1)
            return self.write_infix(t, args), l1
        #elif t in self.variables or t in self.coefficients or t in self.constants or t == 'I':
        elif t in self.variables or t in self.constants or t == 'I':
            return t, expr[1:]
        else:
            val, i = self.parse_int(expr)
            return str(val), expr[i:]

    def prefix_to_infix(self, expr):
        """
        Prefix to infix conversion.
        """
        p, r = self._prefix_to_infix(expr)
        if len(r) > 0:
            raise InvalidPrefixExpression(f"Incorrect prefix expression \"{expr}\". \"{r}\" was not parsed.")
        return f'({p})'

    def rewrite_sympy_expr(self, expr):
        """
        Rewrite a SymPy expression.
        """
        expr_rw = expr
        for f in self.rewrite_functions:
            if f == 'expand':
                expr_rw = sp.expand(expr_rw)
            elif f == 'factor':
                expr_rw = sp.factor(expr_rw)
            elif f == 'expand_log':
                expr_rw = sp.expand_log(expr_rw, force=True)
            elif f == 'logcombine':
                expr_rw = sp.logcombine(expr_rw, force=True)
            elif f == 'powsimp':
                expr_rw = sp.powsimp(expr_rw, force=True)
            elif f == 'simplify':
                expr_rw = simplify(expr_rw, seconds=1)
        return expr_rw

    def infix_to_sympy(self, infix, no_rewrite=False):
        """
        Convert an infix expression to SymPy.
        """
        if not is_valid_expr(infix):
            raise ValueErrorExpression
        expr = parse_expr(infix, evaluate=True, local_dict=self.local_dict)
        if expr.has(sp.I) or expr.has(AccumBounds):
            raise ValueErrorExpression
        if not no_rewrite:
            expr = self.rewrite_sympy_expr(expr)
        return expr

    def _sympy_to_prefix(self, op, expr):
        """
        Parse a SymPy expression given an initial root operator.
        """
        n_args = len(expr.args)
        """
        # derivative operator
        if op == 'derivative':
            assert n_args >= 2
            assert all(len(arg) == 2 and str(arg[0]) in self.variables and int(arg[1]) >= 1 for arg in expr.args[1:]), expr.args
            parse_list = self.sympy_to_prefix(expr.args[0])
            for var, degree in expr.args[1:]:
                parse_list = ['derivative' for _ in range(int(degree))] + parse_list + [str(var) for _ in range(int(degree))]
            return parse_list
        """
        assert (op == 'add' or op == 'mul') and (n_args >= 2) or (op != 'add' and op != 'mul') and (1 <= n_args <= 2)
        
        # square root
        if op == 'pow' and isinstance(expr.args[1], sp.Rational) and expr.args[1].p == 1 and expr.args[1].q == 2:
            return ['sqrt'] + self.sympy_to_prefix(expr.args[0])

        # parse children
        parse_list = []
        for i in range(n_args):
            if i == 0 or i < n_args - 1:
                parse_list.append(op)
            parse_list += self.sympy_to_prefix(expr.args[i])

        return parse_list

    def sympy_to_prefix(self, expr):
        """
        Convert a SymPy expression to a prefix one.
        """
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        elif isinstance(expr, sp.Integer):
            return self.write_int(int(str(expr)))
        elif isinstance(expr, sp.Rational):
            return ['div'] + self.write_int(int(expr.p)) + self.write_int(int(expr.q))
        elif expr == sp.E:
            return ['E']
        elif expr == sp.pi:
            return ['pi']
        elif expr == sp.I:
            return ['I']
        # SymPy operator
        for op_type, op_name in self.SYMPY_OPERATORS.items():
            if isinstance(expr, op_type):
                return self._sympy_to_prefix(op_name, expr)
        # environment function
        #for func_name, func in self.functions.items():
        #    if isinstance(expr, func):
        #        return self._sympy_to_prefix(func_name, expr)
        # unknown operator
        raise UnknownSymPyOperator(f"Unknown SymPy operator: {expr}")

    def reduce_coefficients(self, expr):
        return reduce_coefficients(expr, self.variables.values(), self.coefficients.values())

    def reindex_coefficients(self, expr):
        if self.n_coefficients == 0:
            return expr
        return reindex_coefficients(expr, list(self.coefficients.values())[:self.n_coefficients])

    def extract_non_constant_subtree(self, expr):
        return extract_non_constant_subtree(expr, self.variables.values())

    def simplify_const_with_coeff(self, expr, coeffs=None):
        if coeffs is None:
            coeffs = self.coefficients.values()
        for coeff in coeffs:
            expr = simplify_const_with_coeff(expr, coeff)
        return expr

    def findMaxmumPos(self,pos):
        #in pos, find a substring with a longest length 1
        maxvalue=0
        maxstart=0
        maxend = 0
    
        start=0
        value=0
        for i,e in enumerate(pos):
            #the start point
            if e == 1 and start==0:
                start = i #record the start point
                value=1
            #the last point or the endpoint
            elif (i == len(pos)-1) or (e==0 and start!=0):
                #print("the endpoint")
                if value >maxvalue:
                    maxstart = start
                    maxend = i-1
                    maxvalue=value
                start=0 #renew the start point to 0
            #  accumulate  
            elif e==1 and start!=0:
                value+=1            
            else:
                continue
        return maxstart, maxend, maxvalue
    
    def gen_only_points(self,rng):
        """
        Generate pairs of (function, datapoints)
        start by generating a random function f, and use SymPy to generate datapoints
        """
        #seed = 1 #1 for random seed
        #rng = np.random.RandomState(seed)
        
        x = self.variables['x']
        #x = sp.Symbol('x')    
        #if rng.randint(40) ==0:   #randint (0,40). the probability for true is 1/40 
        #    nb_ops = rng.randint(0,3) # generate a random number from (0,3)
        #else:
        #    nb_ops = rng.randint(3,self.max_ops+1) #(3,16) #the total number of ops  
        nb_ops = rng.randint(1,self.max_ops+1) #only allow 1 or 2 operators
        self.stats = np.zeros(10,dtype = np.int64)       
        #print(nb_ops)
        #try:
            #generate an expression and rewrite it
            #avoid issues in 0 and convert to SymPy
        f_expr = self._generate_expr(nb_ops,self.max_int,rng)
        infix = self.prefix_to_infix(f_expr)
        f = self.infix_to_sympy(infix)
         # skip constant expressions
        if x not in f.free_symbols:
            return None
     
        # generate dataset
        #print(f)
        function = sp.lambdify(x, f)
        data_x = np.linspace(-10,10,self.datalength)
        data_y = np.float32(function(data_x))
  
        pos = big_value_filter(data_y)  
        assert any(pos) is False, "there are nan or big value, so skip it"
        #data = np.array(list(zip(data_x,data_y))).flatten()     
        f_prefix = self.sympy_to_prefix(f)
        return f_prefix,data_y
    
    @timeout(3)
    def gen_func_points(self,rng):
        """
        Generate pairs of (function, datapoints)
        start by generating a random function f, and use SymPy to generate datapoints
        """
        #seed = 1 #1 for random seed
        #rng = np.random.RandomState(seed)
        
        x = self.variables['x']
        #x = sp.Symbol('x')
        
        #if rng.randint(40) ==0:   #randint (0,40). the probability for true is 1/40 
        #    nb_ops = rng.randint(0,3) # generate a random number from (0,3)
        #else:
        #    nb_ops = rng.randint(3,self.max_ops+1) #(3,16) #the total number of ops
        nb_ops = rng.randint(1,self.max_ops+1) #only allow 1 or 2 operators
        self.stats = np.zeros(10,dtype = np.int64)
        
        #print(nb_ops)
        #try:
            #generate an expression and rewrite it
            #avoid issues in 0 and convert to SymPy
        f_expr = self._generate_expr(nb_ops,self.max_int,rng)
        infix = self.prefix_to_infix(f_expr)
        f = self.infix_to_sympy(infix)
        
        numbers = [atom< (self.max_int) for atom in f.atoms() if atom.is_number] 
        #print(f)
        assert all(numbers), "there are number bigger than 10"
        
         # skip constant expressions
        if x not in f.free_symbols:
            return None
        

        # generate dataset
        #print(f)
        function = sp.lambdify(x, f)
        #randomly choose a start point and an end point for the data_x from(-1000,1000)
        start_point = rng.randint(-100,100)
        end_point = rng.randint(start_point+5,100) #there is an at least 5 gap between start and end
        data_x = np.linspace(start_point,end_point,self.datalength)
        data_y = np.float32(function(data_x))
        
        """
        if data contrain inf
        then discard this data(function), because even though we fix this inf problem,
        the dataset still contrain very large number, which can cause nan in model computation
        although we discard this function, that doesn't mean that we'll discard this type of function
        such as exp. exp can still occur in another fuction, without inf
        
        pos not only for nan value, but also for those value who>1e10, 1e10<who<1e-10,and who <-1e10
        """
        pos = big_value_filter(data_y)  
        #if there is any nan value, or big value, we'll shink the x domain to avoid generate nan
        if any(pos):
            #print("encounter nan value abd big value")
            pos=[0 if e else 1 for e in pos]
            start,end,_ = self.findMaxmumPos(pos)
            #print("new domain is:")
            #print(data_x[start],data_x[end])
            data_x = np.linspace(data_x[start],data_x[end],self.datalength)
            data_y = np.float32(function(data_x))
            assert any(np.isnan(data_y)) is False,"still has nan after twice computation"
            assert any((big_value_filter(data_y))) is False, "still has big value"
        
        
        data = np.array(list(zip(data_x,data_y))).flatten()
        # write (data_x,data_y) and f_prefix 
        #data_x and data_y is used as input of model maybe csv
        #prefix is used to model training as output
        f_prefix = self.sympy_to_prefix(f)
        
        #except TimeoutError:
        #    raise
        #except (ValueError, AttributeError, TypeError, OverflowError, NotImplementedError, UnknownSymPyOperator, ValueErrorExpression):
        #    return None
        #except Exception as e:
        #    logger.error("An unknown exception of type {0} occurred in line {1} for expression \"{2}\". Arguments:{3!r}.".format(type(e).__name__, sys.exc_info()[-1].tb_lineno, infix, e.args))
        #    return None
        
        #return f_prefix
        return f_prefix,data
    

    

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument("--operators", type=str, default="add:2,sub:1",
                            help="Operators (add, sub, mul, div), followed by weight")
        parser.add_argument("--max_ops", type=int, default=5,
                            help="Maximum number of operators")
        #parser.add_argument("--max_ops_G", type=int, default=4,
        #                    help="Maximum number of operators for G in IPP")
        parser.add_argument("--max_int", type=int, default=10,#100000
                            help="Maximum integer value")
        parser.add_argument("--int_base", type=int, default=10,
                            help="Integer representation base")
        parser.add_argument("--balanced", type=bool_flag, default=False,
                            help="Balanced representation (base > 0)")
        #parser.add_argument("--precision", type=int, default=10,
        #                    help="Float numbers precision")
        parser.add_argument("--positive", type=bool_flag, default=True,
                            help="Do not sample negative numbers")
        parser.add_argument("--rewrite_functions", type=str, default="",
                            help="Rewrite expressions with SymPy")
        parser.add_argument("--leaf_probs", type=str, default="0.75,0.25,0",
                            help="Leaf probabilities of being a variable, a coefficient, an integer, or a constant.")
        parser.add_argument("--n_variables", type=int, default=1,
                            help="Number of variables in expressions (between 1 and 4)")
        parser.add_argument("--datalength", type=int, default=256,
                            help="the number of data_x and data_y,equals to n_embd*T/2 ")

    def create_train_iterator(self, params, data_path):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator  ...")

        dataset = EnvDataset(
            self,
            train=True,
            rng=None,
            params=params,
            path=(None if data_path is None else data_path[0])
        )
        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 1800),
            batch_size=params.batch_size,
            num_workers=(params.num_workers if data_path is None or params.num_workers == 0 else 1),
            shuffle=False,
            collate_fn=dataset.collate_fn
        )

    def create_test_iterator(self, data_type,params, data_path):
        """
        Create a dataset for this environment.
        """
        assert data_type in ['valid', 'test']
        logger.info(f"Creating {data_type} iterator for ...")

        dataset = EnvDataset(
            self,
            train=False,
            rng=np.random.RandomState(0),
            params=params,
            path=(None if data_path is None else data_path[1 if data_type == 'valid' else 2])
        )
        return DataLoader(
            dataset,
            timeout=0,
            batch_size=params.batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )


class EnvDataset(Dataset):

    def __init__(self, env,train, rng, params, path):
        super(EnvDataset).__init__()
        self.env = env
        self.rng = rng
        self.train = train
        self.batch_size = params.batch_size
        self.env_base_seed = params.env_base_seed
        self.path = path
        self.global_rank = params.global_rank
        self.count = 0
        assert (train is True) == (rng is None)

        # batching
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size
        self.same_nb_ops_per_batch = params.same_nb_ops_per_batch

        # generation, or reloading from file
        if path is not None:
            assert os.path.isfile(path)
            logger.info(f"Loading data from {path} ...")
            with io.open(path, mode='r', encoding='utf-8') as f:
                # either reload the entire file, or the first N lines (for the training set)
                if not train:
                    lines = [line.rstrip().split('|') for line in f]
                    #lines = [line.rstrip() for line in f]
                else:
                    lines = []
                    for i, line in enumerate(f):
                        if i == params.reload_size:
                            break
                        if i % params.n_gpu_per_node == params.local_rank:
                            lines.append(line.rstrip().split('|'))
                            #lines.append(line.rstrip())
            self.data = [xy.split('\t') for _, xy in lines]
            #self.data = [xy.split('\t') for xy in lines]
            self.data = [xy for xy in self.data if len(xy) == 2]
            logger.info(f"Loaded {len(self.data)} equations from the disk.")

        # dataset size: infinite iterator for train, finite for valid / test (default of 5000 if no file provided)
        if self.train:
            self.size = 1 << 60
        else:
            self.size = 5000 if path is None else len(self.data)

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        f, d = zip(*elements)
        nb_ops = [sum(int(word in self.env.OPERATORS) for word in seq) for seq in f]
        # for i in range(len(x)):
        #     print(self.env.prefix_to_infix(self.env.unclean_prefix(x[i])))
        #     print(self.env.prefix_to_infix(self.env.unclean_prefix(y[i])))
        #     print("")
        f = [torch.LongTensor([self.env.word2id[w] for w in seq if w in self.env.word2id]) for seq in f]#x is id
        #y = [torch.LongTensor([self.env.word2id[w] for w in seq if w in self.env.word2id]) for seq in y]#y is float
        #lengths = [len(s) for s in d]
        #print(lengths)
        d = torch.from_numpy(np.array(d)).float() #should use this one
        #when turn it to tensor, generate inf #d = torch.FloatTensor(np.array(d))
        assert any([any(i)for i in torch.isinf(d)]) is False, "has inf in collate_fn"
        f, f_len = self.env.batch_sequences(f)
        d, d_len = self.env.batch_sequences_data(d)
        return (d, d_len), (f, f_len), torch.LongTensor(nb_ops)

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if self.rng is None:
            assert self.train is True
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            self.rng = np.random.RandomState([worker_id, self.global_rank, self.env_base_seed])
            logger.info(f"Initialized random generator for worker {worker_id}, with seed {[worker_id, self.global_rank, self.env_base_seed]} (base seed={self.env_base_seed}).")

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0)
        return 0 if worker_info is None else worker_info.id

    def __len__(self):
        """
        Return dataset size.
        """
        return self.size

    def __getitem__(self, index):
        """
        Return a training sample.
        Either generate it, or read it from file.
        """
        self.init_rng()
        if self.path is None:
            return self.generate_sample()
        else:
            return self.read_sample(index)

    def read_sample(self, index):
        """
        Read a sample.
        """
        if self.train:
            index = self.rng.randint(len(self.data))
        x, y = self.data[index]
        x = x.split()
        y = y.split()
        y = [float(e) for e in y]
        assert len(x) >= 1 and len(y) >= 1
        assert any(np.isinf(y)) is False,"has inf when reading"
        return x, y

    def generate_sample(self):
        """
        Generate a sample.
        """
        while True:

            try:                   
                fd = self.env.gen_func_points(self.rng)
                if fd is None:
                    continue
                f, d = fd
                break
            except TimeoutError:
                continue
            except Exception as e:
                logger.error("An unknown exception of type {0} occurred for worker {4} in line {1} for expression \"{2}\". Arguments:{3!r}.".format(type(e).__name__, sys.exc_info()[-1].tb_lineno, 'F', e.args, self.get_worker_id()))
                continue
        self.count += 1

        # clear SymPy cache periodically
        if CLEAR_SYMPY_CACHE_FREQ > 0 and self.count % CLEAR_SYMPY_CACHE_FREQ == 0:
            logger.warning(f"Clearing SymPy cache (worker {self.get_worker_id()})")
            clear_cache()
        assert any(np.isinf(d)) is False,"still has inf after twice computation"
        return f, d
    
    
#python char_sp.py --operators "add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1"   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language transfer")
    parser.add_argument("--operators", type=str, default="add:2,sub:1",
                        help="Operators (add, sub, mul, div), followed by weight")
    parser.add_argument("--max_ops", type=int, default=5,
                        help="Maximum number of operators")
    #parser.add_argument("--max_ops_G", type=int, default=4,
    #                    help="Maximum number of operators for G in IPP")
    parser.add_argument("--max_int", type=int, default=10,#100000
                        help="Maximum integer value")
    parser.add_argument("--int_base", type=int, default=10,
                        help="Integer representation base")
    parser.add_argument("--balanced", type=bool_flag, default=False,
                        help="Balanced representation (base > 0)")
    #parser.add_argument("--precision", type=int, default=10,
    #                    help="Float numbers precision")
    parser.add_argument("--positive", type=bool_flag, default=True,
                        help="Do not sample negative numbers")
    parser.add_argument("--rewrite_functions", type=str, default="",
                        help="Rewrite expressions with SymPy")
    parser.add_argument("--leaf_probs", type=str, default="0.75,0.25,0",
                        help="Leaf probabilities of being a variable, a coefficient, an integer, or a constant.")
    parser.add_argument("--n_variables", type=int, default=1,
                        help="Number of variables in expressions (between 1 and 4)")
    parser.add_argument("--datalength", type=int, default=256,
                        help="the number of data_x and data_y,equals to n_embd*T/2 ")
    parser.add_argument("--max_len", type=int, default=64,
                        help="the maximum length of expr ")
    params = parser.parse_args()
    
    seed = 1 #1 for random seed
    rng = np.random.RandomState(seed)
    
    env = CharSPEnvironment(params)
    for i in range(10):
        df= env.gen_func_points(rng)
        if df is not None:
            d,f = df
            print(d)
            print(f)
    
    
    