class AbstractChunk:

    provides = set()
    requires = set()

    @classmethod
    def setup_sets(cls, model, *args, **kwargs):
        """Construct the sets this class provides to a model."""

    @classmethod
    def setup_parameters(cls, model, *args, **kwargs):
        """Construct the parameters this class provides to a model."""

    @classmethod
    def setup_variables(cls, model):
        """Construct the variables this class provides to a model."""

    @classmethod
    def setup_objectives(cls, model):
        """Construct the objectives this class provides to a model."""

    @classmethod
    def setup_constraints(cls, model):
        """Construct the constraints this class provides to a model."""


class AbstractModel:

    chunks = []

    @classmethod
    def _test_chunk_order(cls):
        msg = 'Chunk {} has unmet requirements: {}. '
        msg += 'Consider rearranging chunk order.'
        provides = set()
        for chunk in cls.chunks:
            if chunk.requires - provides:
                raise RuntimeError(msg.format(chunk, chunk.requires - provides))
            provides |= chunk.provides

    def setup(self, model, **kwargs):
        self._test_chunk_order()
        self.setup_sets(model, **kwargs)
        self.setup_parameters(model, **kwargs)
        self.setup_variables(model)
        self.setup_objectives(model)
        self.setup_constraints(model)

    def setup_sets(self, model, **kwargs):
        for chunk in self.chunks:
            chunk.setup_sets(model, **kwargs)

    def setup_parameters(self, model, **kwargs):
        for chunk in self.chunks:
            chunk.setup_parameters(model, **kwargs)

    def setup_variables(self, model):
        for chunk in self.chunks:
            chunk.setup_variables(model)

    def setup_objectives(self, model):
        for chunk in self.chunks:
            chunk.setup_objectives(model)

    def setup_constraints(self, model):
        for chunk in self.chunks:
            chunk.setup_constraints(model)
