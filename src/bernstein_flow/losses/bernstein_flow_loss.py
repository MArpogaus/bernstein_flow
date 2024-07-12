# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# %% Author ####################################################################
# file    : bernstein_flow_loss.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2024-07-10 10:13:31 (Marcel Arpogaus)
# changed : 2024-07-12 15:17:17 (Marcel Arpogaus)

# %% License ###################################################################
# Copyright 2020 Marcel Arpogaus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %% Description ###############################################################
"""Custom loss function to calculate the negative logarithmic likelihood with a BNF."""

# %% Imports ###################################################################
from tensorflow import Tensor
from tensorflow.keras.losses import Loss

from bernstein_flow.distributions import BernsteinFlow


# %% Classes ###################################################################
class BernsteinFlowLoss(Loss):
    """NLL of a bijective transformation model using Bernstein polynomials."""

    def __init__(self, **kwargs: dict) -> None:
        """Construct a new instance of the Keras Loss function.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments passed to the super class

        """
        super().__init__(**kwargs)

    def call(self, y: Tensor, pvector: Tensor) -> Tensor:
        """Evaluate the negative logarithmic likelihood given a sample y.

        Parameters
        ----------
        y : Tensor
            A sample.
        pvector : Tensor
            The parameter vector for the normalizing flow.

        Returns
        -------
        Tensor
            Negative logarithmic likelihood.

        """
        flow = BernsteinFlow.new(pvector)
        nll = -flow.log_prob(y)
        return nll
