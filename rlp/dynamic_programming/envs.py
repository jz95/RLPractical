from ..grid_world import GridWorld


class JackCarRentalEnv(FiniteMarkovDecisionProcess):
    """ A demo environment for Jack's Car Rental Problem in page-81.
    """

    def __init__(self, seed):
        FiniteMarkovDecisionProcess.__init__(self, seed)
