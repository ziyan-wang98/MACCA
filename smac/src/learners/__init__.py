
from .nq_learner import NQLearner
from .cq_learner import CQLearner
from .icq_learner import ICQLearner
from .bc_learner import BCLearner
from .omar_learner import OMARLearner
from .baseline_madtkd_learner import MADTKDLearner
from .macca_omar_learner import MACCA_OMARLearner
from .macca_cql_learner import MACCA_CQLearner
from .macca_icq_learner import MACCA_ICQLearner
from .baseline_icq_learner import newICQLearner
from .shaq_learner import SHAQLearner
from .sqddpg_learner import SQDDPGLearner
from .shaq_cql_learner import SHAQCQLLearner

REGISTRY = {}

# Offline MARL Baselines
REGISTRY["nq_learner"] = NQLearner
REGISTRY["cq_learner"] = CQLearner
REGISTRY["bc_learner"] = BCLearner
REGISTRY["icq_learner"] = newICQLearner
REGISTRY["omar_learner"] = OMARLearner
REGISTRY["baseline_madtkd_learner"] = MADTKDLearner
# MACCA 
REGISTRY["macca_omar_learner"] = MACCA_OMARLearner
REGISTRY["macca_icq_learner"] = MACCA_ICQLearner
REGISTRY["macca_cql_learner"] = MACCA_CQLearner
# Credit Assignment Baselines
REGISTRY["shaq_learner"] = SHAQLearner
REGISTRY["sqddpg_learner"] = SQDDPGLearner
REGISTRY["shaq_cql_learner"] = SHAQCQLLearner