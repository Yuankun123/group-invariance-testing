from typing import Callable, Generator, Generic, TypeVar
import math

_G = TypeVar("_G")
_X = TypeVar("_X")


class SampleSpace(Generic[_X]):
    """Sample space containing elements of type _X."""

    def __init__(self, element_type):
        super().__init__()
        self.element_type = element_type

    def is_element(self, x) -> bool:
        raise NotImplementedError


class ActionGroup(Generic[_G, _X]):
    """A group acting on a given sample space."""

    def action(self, g: _G, x: _X) -> _X:
        raise NotImplementedError

    def elements(self) -> Generator[_G, None, None]:
        """Return the elements. Implemented for finite groups."""
        raise NotImplementedError

    def random_elem(self, rng) -> _G:
        raise NotImplementedError

    def orbit_average(self, f: Callable[[_G], float]) -> float:
        """Compute the orbit average of a positive function over the group."""
        raise NotImplementedError


class DataGenerator(Generic[_X]):
    def __init__(self, space: SampleSpace[_X]):
        self.space = space

    def get_random_elem(self, rng) -> _X:
        raise NotImplementedError


class DynamicT(Callable[[_X], float]):
    """Callable positive score used inside the post-hoc p-process."""

    def __call__(self, _x) -> float:
        raise NotImplementedError


class LogEvidenceUpdater(Generic[_G, _X]):
    """Maintain observations and sequentially accumulate log evidence."""

    def __init__(self):
        self.observations: list[_X] = []
        self.current_log_evidence = 0

    def update(self, G: ActionGroup[_G, _X], x: _X, rng):
        self.observations.append(x)

    def get_log_evidence(self, group: ActionGroup[_G, _X], x: _X, rng):
        raise NotImplementedError

    def one_step(self, G: ActionGroup[_G, _X], x: _X, rng):
        self.current_log_evidence += self.get_log_evidence(G, x, rng)
        self.update(G, x, rng)
        return self.current_log_evidence

    @staticmethod
    def from_dynamic_T(DyT):
        """Wrap a DynamicT factory as a finite-group log-evidence updater."""

        class _FromDynamicT(LogEvidenceUpdater):
            def get_log_evidence(self, G: ActionGroup[_G, _X], x: _X, rng):
                T: DynamicT = DyT(self.observations)

                def orbit_eval(g: _G) -> float:
                    value = T(G.action(g, x))
                    assert value > 0, "T_i must be strictly positive."
                    return value

                test_value = T(x)
                assert test_value > 0, "T_i must be strictly positive."
                return math.log(test_value) - math.log(G.orbit_average(orbit_eval))

        return _FromDynamicT()


class SequentialTest(Generic[_G, _X]):
    def __init__(
        self,
        alpha: float,
        group: ActionGroup[_G, _X],
        data: DataGenerator[_X],
        log_evi: LogEvidenceUpdater[_G, _X],
        step=100,
        rng=None,
    ):
        assert 0 < alpha <= 1, "Alpha must be in the range (0, 1]."
        self.alpha = alpha
        self.G = group
        self.data = data
        self.log_evi = log_evi
        self.step = step
        self.rng = rng

    def run(self) -> bool:
        for _ in range(self.step):
            new_obs = self.data.get_random_elem(self.rng)
            self.log_evi.one_step(self.G, new_obs, self.rng)
            if self.log_evi.current_log_evidence >= math.log(1 / self.alpha):
                return True
        return False
