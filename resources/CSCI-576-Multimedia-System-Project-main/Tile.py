class Tile:
    def __init__(self, corners=None, edges=None, image=None, rotation=0):
        if edges is None:
            edges = []
        if corners is None:
            corners = []
        self.corners = corners
        self.edges = edges
        self.image = image  
        self.rotation = rotation
        self.position = None 

        self.initial_position = (0, 0)  # (x, y) for the center of tile
        self.final_position = (0, 0)
        self.initial_rotation = 0
        self.final_rotation = 0
        self.mask = None

        @property
        def x(self):
            return self.position[0]

        @property
        def y(self):
            return self.position[1]

        @x.setter
        def x(self, v):
            self.position = (v, self.position[1])

        @y.setter
        def y(self, v):
            self.position = (self.position[0], v)