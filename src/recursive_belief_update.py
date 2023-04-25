class RBU:
    def __init__(self, x_range, y_range, grid_size=0.20, alpha=0.7, beta=0.9):
        self.x_range = x_range
        self.y_range = y_range
        self.grid_size = grid_size
        self.alpha = alpha
        self.beta = beta
        self.safety_belief = {}

        x = round(x_range[0], 4)
        y = round(y_range[0], 4)
        while x <= x_range[1]:
            y = round(y_range[0], 4)
            while y <= y_range[1]:
                self.safety_belief[(x, y)] = 0.0
                y = round(y + self.grid_size, 4)
            x = round(x+self.grid_size, 4)

    def update(self, visible_locs: dict):
        notVisible = set(self.safety_belief.keys()) - set(visible_locs.keys())
        for loc in visible_locs.keys():
            self.safety_belief[loc] = self.alpha*visible_locs[loc] + (1 - self.alpha)*self.safety_belief[loc]
        for loc in notVisible:
            self.safety_belief[loc] = self.beta*self.safety_belief[loc]

    def bring_inside_range(self, x, y):
        x = min(max(x, self.x_range[0]), self.x_range[1])
        y = min(max(y, self.y_range[0]), self.y_range[1])
        return x, y

    def round_to_nearest_grid_cell(self, x, y):
        x = round(x/self.grid_size)*self.grid_size
        y = round(y/self.grid_size)*self.grid_size
        return x, y

    def round_and_range(self, x, y):
        x, y = self.round_to_nearest_grid_cell(x, y)
        x, y = self.bring_inside_range(x, y)
        return round(x, 4), round(y, 4)
