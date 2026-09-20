import warnings

refractive_indexes = {"Air": 1.0003, "Glass": 1.5, "Acrylic": 1.48899, "Water": 1.34}


class Layer:
    def __init__(self, height, material=None, refractive_index=None):
        self.height = height
        if refractive_index is None:
            if material is not None:
                if material in refractive_indexes.keys():
                    self.refractive_index = refractive_indexes[material]
                else:
                    raise KeyError(f"Dont know {material}, provide refractive index.")
            if material is None:
                raise ValueError("Enter either a refractive index or material for the layer.")
        elif refractive_index is not None:
            if material is None:
                self.refractive_index = refractive_index
            if material is not None:
                self.refractive_index = refractive_index
                warnings.warn("Material and refractive index provided separately, you should choose only one way to determine the layer.")

    def effective_height(self, fluid_refractive_index):
        return fluid_refractive_index * self.height / self.refractive_index
