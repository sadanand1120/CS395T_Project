class SafetyBelief:
    def __init__(self, map, alpha=0.7, beta=0.9):
        self.map = map
        self.alpha = alpha
        self.beta = beta

        #Initialize belief to 0 for all map locations
        self.safety_belief = {}
        for loc in map:
            self.safety_belief[loc] = 0.0

    def update(self, new_safety_estimate):
        #Get visible locations -- assumes new safety estimate will be a dict
        visible_locs = []
        for loc in new_safety_estimate:
            visible_locs.append(loc)

        for loc in self.map:
            if loc in visible_locs:
                self.safety_belief[loc] = self.alpha*new_safety_estimate[loc] + (1 - self.alpha)*self.safety_belief[loc]
            else:
                self.safety_belief[loc] = self.beta*self.safety_belief[loc]