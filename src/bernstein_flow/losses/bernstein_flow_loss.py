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
# modified time : 2020-09-11 17:11:03
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
import tensorflow as tf
from tensorflow.keras.losses import Loss

from bernstein_flow.distributions import BernsteinFlow


class BernsteinFlowLoss(Loss):
    """
    This class describes a normalizing flow using Bernstein polynomials as
    Keras Loss function
    """

    def __init__(
            self,
            M: int,
            **kwargs: dict):
        """
        Constructs a new instance of the Keras Loss function.

        :param      M:       Order of the used Bernstein polynomial bijector.
        :type       M:       int
        :param      kwargs:  Additional keyword arguments passed to the supper
                             class
        :type       kwargs:  dictionary
        """
        self.bernstein_flow = BernsteinFlow(M)
        super().__init__(**kwargs)

    def call(self,
             y: tf.Tensor,
             pvector: tf.Tensor) -> tf.Tensor:
        """
        Evaluates the negative logarithmic likelihood given a sample y.

        :param      y:        A sample.
        :type       y:        Tensor
        :param      pvector:  The parameter vector for the normalizing flow.
        :type       pvector:  Tensor

        :returns:   negative logarithmic likelihood
        :rtype:     Tensor
        """
        flow = self.bernstein_flow(pvector)

        nll = -flow.log_prob(y)

        return nll
