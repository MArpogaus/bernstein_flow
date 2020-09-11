#!env python3
# -*- coding: utf-8 -*-
# AUTHOR INFORMATION ##########################################################
# file   : bernstein_flow_loss.py
# brief  : [Description]
#
# author : Marcel Arpogaus
# date   : 2020-03-31 19:22:59
# COPYRIGHT ###################################################################
# NEEDS TO BE DISCUSSED WHEN RELEASED!
#
# PROJECT DESCRIPTION #########################################################
#
# NOTE: this project is following the PEP8 style guide
#
# Bla bla...
#
# CHANGELOG ##################################################################
# modified by   : Marcel Arpogaus
# modified time : 2020-09-11 15:25:00
#  changes made : using JointDistributionSequential
# modified by   : Marcel Arpogaus
# modified time : 2020-03-31 19:22:59
#  changes made : newly written
###############################################################################

# PYTHON AUTHORSHIP INFORMATION ###############################################
# ref.: https://stackoverflow.com/questions/1523427

"""baseline.py: [Description]"""

__author__ = ["Marcel Arpogaus"]
# __authors__ = ["author_one", "author_two" ]
# __contact__ = "kontakt@htwg-konstanz.de"

# __copyright__ = ""
# __license__ = ""

__date__ = "2020-03-31 19:22:59"
# __status__ = ""
# __version__ = ""

# REQUIRED PYTHON MODULES #####################################################
from tensorflow.keras.losses import Loss

from ..distributions import BernsteinFlow


class BernsteinFlowLoss(Loss):
    def __init__(
            self,
            M,
            **kwargs):
        self.bernstein_flow = BernsteinFlow(M)
        super().__init__(**kwargs)

    def call(self, y, pvector):

        flow = self.bernstein_flow(pvector)

        nll = -flow.log_prob(y)

        return nll
