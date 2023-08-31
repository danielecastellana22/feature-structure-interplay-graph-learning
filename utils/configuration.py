from .serialisation import from_yaml_file, from_json_file, to_json_file, to_yaml_file
from .misc import string2class
import copy


class Config(dict):

    def __init__(self, **config_dict):
        super().__init__()

        # set attributes
        for k, v in config_dict.items():
            if isinstance(v, dict):
                # if is dict, create a new Config obj
                v = Config(**v)
            super(Config, self).__setitem__(k, v)

    # the dot works as []
    def __getattr__(self, item):
        if item in self:
            return self.__getitem__(item)
        else:
            raise AttributeError('The key {} must be specified!'.format(item))

    def __setattr__(self, key, value):
        self[key] = value

    def __add__(self, other):
        return Config(**self, **other)

    @classmethod
    def from_json_fle(cls, path):
        return cls(**from_json_file(path))

    @classmethod
    def from_yaml_file(cls, path):
        return cls(**from_yaml_file(path))

    def to_dict(self):
        d = {}
        for k, v in self.items():
            if isinstance(v, Config):
                d[k] = v.to_dict()
            else:
                d[k] = v
        return d

    def to_json_fle(self, path):
        to_json_file(self.to_dict(), path)

    def to_yaml_file(self, path):
        to_yaml_file(self.to_dict(), path)

    @staticmethod
    def __is_in_grid_search__(k, v):
        return (isinstance(v, list) and not k.endswith('_list')) or \
               (isinstance(v, list) and k.endswith('_list') and isinstance(v[0], list))

    def build_config_grid(self):

        def __rec_build__(d, k_list, d_out):
            if len(k_list) == 0:
                return [copy.deepcopy(d_out)]
            out_list = []
            k = k_list[0]
            v = d[k]
            if isinstance(v, dict):
                # now becomes a list
                v = __rec_build__(v, list(v.keys()), Config())

            if Config.__is_in_grid_search__(k, v):
                for vv in v:
                    d_out[k] = vv
                    out_list += __rec_build__(d, k_list[1:], d_out)
            else:
                d_out[k] = v
                out_list += __rec_build__(d, k_list[1:], d_out)

            return out_list

        return __rec_build__(self, list(self), Config())

    def get_grid(self):

        def __get_grid_recursive__(config_dict):
            ris = {}
            for k, v in config_dict.items():
                if Config.__is_in_grid_search__(k,v):
                    # this is a hyperparameter in the grid search
                    ris[k] = v
                elif isinstance(v, dict):
                    # is another dict
                    for kk, vv in __get_grid_recursive__(v).items():
                        ris[k+'.'+kk] = vv
            return ris

        return __get_grid_recursive__(self)


def create_object_from_config(obj_config, **other_params):
    class_name = string2class(obj_config['class'])
    params = obj_config['params'] if 'params' in obj_config else {}
    all_params = copy.deepcopy(params)
    all_params.update(other_params)
    return class_name(**all_params)