from abc import abstractmethod

from ..util.categories import category


class BaseNode:
    name = "Node"
    display_name = name

    @classmethod
    @abstractmethod
    def INPUT_TYPES(cls):
        """
        see https://docs.comfy.org/custom-nodes/backend/datatypes

        return {
            "required": {},
            "optional": {},
            "hidden": {}
        }
        """
        return {
            "required": {}, # (type, options)
            "optional": {},
            "hidden": {}
        }  # required, optional, hidden

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = ""
    CATEGORY = category("misc")