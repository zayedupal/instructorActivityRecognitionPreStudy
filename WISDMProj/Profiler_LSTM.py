import WISDM_LSTM_Model
from memory_profiler import profile,LogFile
import cProfile

@profile
def profile():
    WISDM_LSTM_Model.main()

cProfile.run("profile()")