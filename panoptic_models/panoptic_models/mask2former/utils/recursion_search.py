# import packages
from functools import reduce
from typing import Union, Any, Tuple, List
import operator
import numpy as np
from copy import deepcopy


def type_modifier(output_dict: dict) -> dict:
    """
    Changes every type 'int64' in the dict to 'int

    Parameters
    ----------
    output_dict         - input dict

    Returns
    -------
    output_dict         - adjusted output dict
    """

    def get_by_path(root: dict, items: list) -> Any:
        """Access a nested object in root by item sequence."""
        return reduce(operator.getitem, items, root)

    def set_by_path(root: dict, items: list, value: Any) -> dict:
        """Set a value in a nested object in root by item sequence."""
        get_by_path(root, items[:-1])[items[-1]] = value
        return root

    def recursion_search(
        document: Union[dict, str], key_list: list, copy_dictionary
    ) -> Tuple[dict, List[str]]:
        """
        Recursive function to modified key values in the dict
        """
        if isinstance(document, dict):
            for key, value in document.items():
                key_list.append(key)
                key_list, copy_dictionary = recursion_search(
                    document=value, key_list=key_list, copy_dictionary=copy_dictionary
                )
                key_list = key_list[:-1]

        elif isinstance(document, list):
            for idx, entry in enumerate(document):
                key_list.append(idx)
                key_list, copy_dictionary = recursion_search(
                    document=entry, key_list=key_list, copy_dictionary=copy_dictionary
                )
                key_list = key_list[:-1]

        elif type(document).__name__ == "int64":
            copy_dictionary = set_by_path(copy_dictionary, key_list, int(document))

        return key_list, copy_dictionary

    _, modified_dict = recursion_search(
        document=output_dict, key_list=list([]), copy_dictionary=deepcopy(output_dict)
    )

    return modified_dict


def main():
    """
    Test implementation
    """
    key_to_change = np.array([3], dtype=np.int64)
    nested = {
        "a": [{"b": 1, "c": 2}],
        "d": "hallo",
        "e": {"f": [{"g": 4, "h": key_to_change[0]}, {"i": 6, "j": 7}]},
    }
    nested_modified = type_modifier(nested)
    assert type(nested_modified["e"]["f"][0]["h"]).__name__ == "int"


if __name__ == "__main__":
    main()
