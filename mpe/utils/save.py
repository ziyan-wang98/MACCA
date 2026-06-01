import os
import numpy as np

class Saver:


    def __init__(self, args):
        self.args = args
        self.info = []
        if args.save_rews:
            make_dir('log/rews', clear=False)
            # self.rews_record = {}
            # self.rews_record[args.env] = []
            self.rews_record = [[]]
            self.rews_record_cycle = []
            
    def make_dir(dir_name, clear=True):
        if os.path.exists(dir_name):
            if clear:
                try: shutil.rmtree(dir_name)
                except: pass
                try: os.makedirs(dir_name)
                except: pass
        else:
            try: os.makedirs(dir_name)
            except: pass

    def save_cs(self, agent, t):
        # import pdb; pdb.set_trace()
        if self.args.algorithm_name == "causal_omar" or self.args.algorithm_name == "causal_cql":
            causal_structure = agent.get_structure()
            for key, value in causal_structure.items():
                log_folder = 'log/causal_structure/' + self.args.env_id + '/' +self.args.algorithm_name + '/' + self.args.tag + self.args.timestamp
                os.makedirs(log_folder, exist_ok=True)
                np.save(log_folder + '/' + str(t) + key + '.npy', value.cpu().detach().numpy())
