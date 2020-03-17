import cProfile

import WISDM_CNN_Model
from memory_profiler import profile


@profile
def profile():
    WISDM_CNN_Model.main()

cProfile.run("profile()")

