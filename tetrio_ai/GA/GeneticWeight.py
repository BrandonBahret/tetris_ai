import numpy as np

from Types.NumbaDefinitions import GeneticWeightBundle


class GeneticWeight:
    def __init__(self, minimum, maximum, step=0.01):
        self.__value_range = (float(minimum), float(maximum))
        self.__value_space = np.round(np.arange(minimum, maximum+step, step), decimals=4)
        self.__value_space_size = len(self.__value_space)
        self.__value = np.random.choice(self.__value_space)
        # self.__value = (maximum - minimum) * np.random.sample() + minimum

    def __repr__(self):
        return self.to_bundle().str(flatten=True)

    def to_bundle(self):
        minimum, maximum = self.__value_range
        value = self.__value
        return GeneticWeightBundle.new(minimum, maximum, value)

    @property
    def value_space(self):
        return self.__value_space

    @property
    def value_range(self):
        return self.__value_range

    @property
    def value(self):
        return self.__value

    def set_weight(self, value):
        self.__value = value
        return self

    def mutate(self):
        # minimum, maximum = self.__value_range
        # self.__value = (maximum - minimum) * np.random.sample() + minimum
        self.__value = np.random.choice(self.__value_space)

    # def mutate(self):
    #     vspace_size = self.__value_space_size
    #     vspace = self.__value_space

    #     value_i = np.where(vspace == self.__value)[0].item()
    #     random_off = np.random.randint(0, max(2, vspace_size//2))
        
    #     self.__value = vspace[(value_i + random_off) % vspace_size]
