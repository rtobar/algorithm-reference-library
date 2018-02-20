""" Algorithm Reference Library

"""

import os
if 'ARL_USE_DLG_DELAYED' in os.environ:
    from dlg import delayed
else:
    from dask import delayed  # @Reimport
del os