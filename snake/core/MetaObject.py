import json

class MetaObject(object):
    """
    Utilitaire pour transformer un dictionnaire
    en object python. Les clefs deviennent des
    attributs.
    """
    def __init__(self, attributes):
        MetaObject.override_from_dict(self, attributes)

    @classmethod
    def from_dict(cls, attributes):
        return MetaObject(attributes)

    @classmethod
    def from_kwargs(cls, **kwargs):
        return cls.from_dict(kwargs)

    @classmethod
    def from_json(cls, filename):
        try:
            with open(filename) as f:
                return json.load(f,
                                 object_hook=cls.from_dict)
        except Exception as e:
            print(e.message)
            return None

    @classmethod
    def override_from_kwargs(cls, instance, **kwargs):
        cls.override_from_dict(instance, kwargs)

    @classmethod
    def override_from_dict(cls, instance, attributes):
        if isinstance(attributes, dict):
            if isinstance(instance, MetaObject):
                # initialiser un MetaObject
                instance.__dict__.update(attributes)
            else:
                # overrider objet complexe
                attr_keys = set(attributes.keys())
                inst_keys = set(instance.__dict__.keys())

                new_keys = attr_keys.difference(inst_keys)
                new_dict = {k:attributes[k] for k in new_keys}
                instance.__dict__.update(new_dict)

                # clefs communes
                common_dict = {}
                for k in attr_keys.intersection(inst_keys):
                    inst_v = instance.__dict__[k]
                    attr_v = attributes[k]
                    if isinstance(inst_v, dict):
                        # dict recursif
                        inst_v.update(vars(attr_v))
                    else:
                        common_dict[k] = attr_v

                instance.__dict__.update(common_dict)
        else:
            raise RuntimeError("MetaObject can only "
                               "be constructed from dict")

    @classmethod
    def override_from_json(cls, instance, filename):
        try:
            with open(filename) as f:
                attributes = json.load(f)
        except Exception as e:
            print(e.message)
            return None
        else:
            return cls.override_from_dict(instance, attributes)

    @classmethod
    def override_from_object(cls, instance, object):
        cls.override_from_dict(instance, vars(object))
