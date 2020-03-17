import WISDM_LRCN_Model
from memory_profiler import profile,LogFile
import cProfile

@profile
def profile():
    WISDM_LRCN_Model.main()

cProfile.run("profile()")