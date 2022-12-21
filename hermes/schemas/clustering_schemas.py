from typing import Union, Type
from pydantic import BaseModel
from hermes.similarity import BaseSimilarity
from hermes.distance import BaseDistance


class PrivateAttr(BaseModel):
    calculated: bool = False  # has been set?
    metric_type: Union[Type[BaseSimilarity], Type[BaseDistance]]  # what type?

    def __bool__(self):
        return self.calculated
