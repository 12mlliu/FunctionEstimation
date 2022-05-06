"""
In this file, I will comb the environment 

"""
#sympy is a library for symbolic mathematics
import sympy as sp

#main.py
env = build_env(params)

#__init__.py

ENVS ={'char_sp':CharSPEnvironment}

def buil_env(params):
    env = ENVS[params.env_name](params) #env_name ???
    tasks = # task (prim_fwd, prim_bwd, prim_ibp, ode1, ode2)
    return env

#char_sp.py
#functions gen_prim_fwd, gen_prim_bwd, gen_prim_ibp, gen_ode1, gen_ode2 
#responsiible for the generation of the 5 tasks, inside the CharSPEnvironment class
#if want to try a new task, just need to add a new function
class CharSPEnvironment(object):
 
    TRAINING_TASKS = {'prim_fwd', 'prim_bwd', 'prim_ibp', 'ode1', 'ode2'}
    
    SYMPY_OPERATORS = {
    # Elementary functions
          # Elementary functions
        sp.Add: 'add',
        sp.Mul: 'mul',
    }
    
    OPERATORS ={
        #operators and thier classification, 1 means uniary operator, 2 means 2 operators
     # operators with (unnormalized sampling probabilities)
        'add': 2,
        'sub': 2,
        'pow2': 1,
        'pow3': 1,
    }
    def __init__(self,params):
        
        self.max_ops =15 #maximum number of operators at generation
        self.max_int 5 #max value of sampled integers
        self.positive = True #sign of sampled integers
        self.max_len = 512#maximum length of generated equations
        
        #parse the operators and thier weight
        #params.operators ="add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2"
        #we will classifiy these operators into uniary operators and binary operator and normilize them
        ops = params.operators.split(',')
        ops = sorted([x.split(':') for x in ops])
        
        self.all_ops =[o for o, _ in ops]
        self.uniary_ops = [o for o, _ in ops if self.OPERATORS[o]==1]
        self.binary_ops = [o for o,_ in ops  if self.OPERATORS[o]==2]
        self.all_ops_probs = np.array([float(w) for _,w in ops]).astype(np.float64)
        self.all_ops_probs = self.all_ops_prob/self.all_ops_prob.sum()
        
        #symbols / element
        self.constants = ['pi','E']
        self.variables = OrderedDict({
            'x': sp.Symbol('x', real=True, nonzero=True),  # , positive=True
            'y': sp.Symbol('y', real=True, nonzero=True),  # , positive=True
            'z': sp.Symbol('z', real=True, nonzero=True),  # , positive=True
            't': sp.Symbol('t', real=True, nonzero=True),  # , positive=True
        })
        #a1 a2
         self.coefficients = OrderedDict({
            f'a{i}': sp.Symbol(f'a{i}', real=True)
            for i in range(10)
        })
        self.functions = OrderedDict({
            'f': sp.Function('f', real=True, nonzero=True),
            'g': sp.Function('g', real=True, nonzero=True),
            'h': sp.Function('h', real=True, nonzero=True),
        self.symbols = ['I', 'INT+', 'INT-', 'INT', 'FLOAT', '-', ]
        #self.balanced:
        if balanced:
            self.elements = ['-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4']
        else:
            self.elements = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            
        #sympy elements  
        local_dict ={} # combination of self.variables self.coefficients self.functions
            
        #are this going to be predicted
        #sympy vocabulary
        #SPECIAL_WORDS = ['<s>', '</s>', '<pad>', '(', ')']
        #SPECIAL_WORDS = SPECIAL_WORDS + [f'<SPECIAL_{i}>' for i in range(len(SPECIAL_WORDS), 10)]
        self.words  = SPECIAL_WORDS + self.constants +list(self.variables.keys())
        self.id2word = {i:w for i,w in enumerate(self.words)}
        self.word2id = {w:i for i,w in enumerate(self.words)}
            
            #number of words
            self.n_words = params.n_words =len(self.words)
            
            #indices why these two index are not inserted in self.words
            self.eos_index = params.eos_index =0
            self.pad_index = params.pad_index =1
        
            #leaf probability --leaf_probs "0.75,0,0.25,0"   # leaf sampling probabilities (x,y,z,t)
            #variables(x,y,z,t)  + coefficients(a1,a2,a3) + integers + constants(e,pi)
            params.self.n_variables =1 #number of variables (x, y, z)
            params.self.n_coefficients =0 #number of coefficients (a_0,a_1..)
            params.self.max_init =5 # max value of sampled integers
            params.positive = true #sign of sampled integers
            #if self.leaf_prob[3] >0  + len(self.constants)
            self.n_leaves = self.n_variables + self.n_codfficients + self.max_int +self.constants
            
            #generation parameters
            self.nl =1 #self.n_leaves whose distribution probability kept in self.leaf_prob[0.75,0,0.25,0]
            self.p1 =1 #len(self.una_ops)
            self.p2 =2 #len(self.bin_ops)
            
            #initialize distribution for binary and uniary-binary trees
            self.bin_dist = self.generate_bin_dist(params.max_ops)
            self.ubi_dist = self.generate_ubi_dist(params.max_ops)
            
            #rewritw expressions
            
            
      def gen_prim_fwd(self.rng):
            """
            Generate paris of (function, primitive)
            start by generating a random function f, and use SymPy to compute F
            """
            
            x = self.variables['x'] #sp.Symbol('x', real=True, nonzero=True),
            try:
            #generate an expression and rewrite it,
            #avoid issues in 0 and convert to SymPy
            
     
       def init_rng (self):
            """
            initialize random generator for training
       
            """
            if self.rng is None:
                self.rng = np.random.RandomState([worker_id,self.global_rank,self.env_base_seed])
            
       def generate_sample(self):
            """
            generate a sample
            """
            
            xy = self.env.gen_prim_fwd(self.rng)
            x,y =xy
            return x, y