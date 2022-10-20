class Recipe(object):
    """Light-weight class that defines the configs to launch types of
    jobs. These jobs are defined for all datasets given by the datasets
    property.

    Args:
        name (string): The name of the config
        config_mixins (list[string]): Name of the human-readable configs
        base: (string): The base config this runs off of.
    """
    def __init__(self, name, config_mixins, base="default-base"):
        self._name = name
        self._config_mixins = config_mixins
        self._base = base

    @property
    def config_mixins(self):
        return self._config_mixins

    @property
    def datasets(self):
        return ["alchemy", "tangrams", "scene", "alchemy-multi-step", "tangrams-multi-step", "scene-multi-step"]

    @property
    def base(self):
        return self._base

    @property
    def name(self):
        return self._name

    def __str__(self):
        return 'Recipe({}: {} + {})'.format(
            self.name, self.base, self.config_mixins)
    __repr__ = __str__


class AlchemyRecipe(Recipe):
    @property
    def datasets(self):
        return ["alchemy"]


class TangramsRecipe(Recipe):
    @property
    def datasets(self):
        return ["tangrams"]


class SceneRecipe(Recipe):
    @property
    def datasets(self):
        return ["scene"]


class Cookbook(object):
    """A collection of recipes"""
    def __init__(self, recipes):
        self._recipes = recipes

    @property
    def recipes(self):
        return self._recipes


class RLongCookbook(Cookbook):
    def __init__(self):
        self._recipes = [
            # Baseline
            Recipe(name="default", config_mixins=[]),

            # Alpha (q_RL)
            Recipe(name="alpha=0", config_mixins=["alpha=0"]),

            # Beta
            Recipe(name="beta=0", config_mixins=["beta=0"]),
            Recipe(name="beta=0.25", config_mixins=["beta=0.25"]),
            #Recipe(name="beta=0.5", config_mixins=["beta=0.5"]),
            #Recipe(name="beta=0.75", config_mixins=["beta=0.75"]),

            # Beam search
            Recipe(name="beam-32", config_mixins=["beam-search"]),
            Recipe(name="beam-128", config_mixins=["beam-search", "train_beam_size=128"]),

            # Particle Filtering
            #Recipe(name="particle-filtering",
            #       config_mixins=["train_beam_size=256", "particle-filtering"]),

            # Epsilon
            Recipe(name="epsilon=0.05", config_mixins=[
                "beam-search", "epsilon=0.05"]),
            #Recipe(name="epsilon=0.08", config_mixins=[
            #    "beam-search", "epsilon=0.08"]),
            #Recipe(name="epsilon=0.1", config_mixins=[
            #    "beam-search", "epsilon=0.1"]),
            #Recipe(name="epsilon=0.12", config_mixins=[
            #    "beam-search", "epsilon=0.12"]),
            Recipe(name="epsilon=0.25", config_mixins=[
                "beam-search", "epsilon=0.25"]),

            # REINFORCE
            Recipe(name="reinforce+beam=001+noahead", config_mixins=[
                "batched-reinforce-basic", "batched-reinforce-epsilon=0.2",
                "train_beam_size=1"]),
            Recipe(name="reinforce+beam=032+noahead", config_mixins=[
                "batched-reinforce-basic", "batched-reinforce-epsilon=0.2"]),
            #Recipe(name="reinforce+beam=128+noahead", config_mixins=[
            #    "batched-reinforce-basic", "batched-reinforce-epsilon=0.2",
            #    "train_beam_size=128"]),
            #Recipe(name="reinforce+beam=001+lookahead", config_mixins=[
            #    "batched-reinforce-basic", "batched-reinforce-epsilon=0.2",
            #    "train_beam_size=1", "batched-reinforce-lookahead"]),
            #Recipe(name="reinforce+beam=032+lookahead", config_mixins=[
            #    "batched-reinforce-basic", "batched-reinforce-epsilon=0.2",
            #    "batched-reinforce-lookahead"]),
            #Recipe(name="reinforce+beam=128+lookahead", config_mixins=[
            #    "batched-reinforce-basic", "batched-reinforce-epsilon=0.2",
            #    "train_beam_size=128", "batched-reinforce-lookahead"]),

            # REINFORCE + baseline
            Recipe(name="reinforce+baseline=0.1", config_mixins=[
                "batched-reinforce-basic", "batched-reinforce-epsilon=0.2",
                "baseline=0.1"]),
            #Recipe(name="reinforce+baseline=0.03", config_mixins=[
            #    "batched-reinforce-basic", "batched-reinforce-epsilon=0.2",
            #    "baseline=0.03"]),
            Recipe(name="reinforce+baseline=0.01", config_mixins=[
                "batched-reinforce-basic", "batched-reinforce-epsilon=0.2",
                "baseline=0.01"]),
            #Recipe(name="reinforce+baseline=0.003", config_mixins=[
            #    "batched-reinforce-basic", "batched-reinforce-epsilon=0.2",
            #    "baseline=0.003"]),
            Recipe(name="reinforce+baseline=0.001", config_mixins=[
                "batched-reinforce-basic", "batched-reinforce-epsilon=0.2",
                "baseline=0.001"]),
            Recipe(name="reinforce+baseline=0.0001", config_mixins=[
                "batched-reinforce-basic", "batched-reinforce-epsilon=0.2",
                "baseline=0.0001"]),
            Recipe(name="reinforce+baseline=0.00001", config_mixins=[
                "batched-reinforce-basic", "batched-reinforce-epsilon=0.2",
                "baseline=0.00001"]),

            # REINFORCE + logistic baseline
            Recipe(name="reinforce+logistic-value-fxn", config_mixins=[
                    "batched-reinforce-basic", "batched-reinforce-epsilon=0.2",
                    "logistic_value_fxn"]),

            # History (h), Stack (s), Independent Utterance (IU)
            Recipe(name="stack", config_mixins=["only-use-stack-emb"]),
            #Recipe(name="h+s", config_mixins=["stack-emb"]),
            #Recipe(name="iu", config_mixins=["indep-utt-expl"]),
            #Recipe(name="h+s+iu", config_mixins=["stack-emb", "indep-utt-expl"]),

            # Multi-step training
            Recipe(name="multi-step-train", config_mixins=["multi-step-train"]),

            # Best
            AlchemyRecipe(name="alchemy-best", config_mixins=[
                "beta=0", "only-use-stack-emb"]),
            TangramsRecipe(name="tangrams-best", config_mixins=["beta=0.25"]),
            SceneRecipe(name="scene-best", config_mixins=[
                "beta=0", "only-use-stack-emb"]),
        ]

    def get_recipe_name(self, configs, base):
        for recipe in self._recipes:
            if sorted(recipe.config_mixins) == sorted(configs) and recipe.base == base:
                return recipe.name
        return None
