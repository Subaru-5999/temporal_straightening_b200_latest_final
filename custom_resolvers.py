import hydra
from omegaconf import OmegaConf

@hydra.main(config_path=None)
def register_resolvers(cfg):
    pass

# Define the resolver function
def replace_slash(value: str) -> str:
    return value.replace('/', '_')

def replace_substring(value: str, old: str, new: str) -> str:
    return str(value).replace(str(old), str(new))

# (lambda_cf, ccr_rho, ccr_action_source, mca_weight)
CCR_TAG_DEFAULTS = (0.0, 0.0, "synthetic", 0.0)

def _fmt_num(value: float) -> str:
    # '0.1' -> '0p1', '0.05' -> '0p05', '1e-05' -> '1e-05'; keeps paths free of '.'
    return f"{float(value):g}".replace(".", "p")

def ccr_tag(lambda_cf, rho, action_source, mca_weight) -> str:
    # A missing key (e.g. an older config resolved through `oc.select` without a
    # fallback) arrives as None; treat it as the default entry of CCR_TAG_DEFAULTS
    # so no separate literal can drift away from conf/train.yaml.
    given = (lambda_cf, rho, action_source, mca_weight)
    filled = tuple(d if v is None else v for v, d in zip(given, CCR_TAG_DEFAULTS))
    values = (float(filled[0]), float(filled[1]), str(filled[2]), float(filled[3]))
    if values == CCR_TAG_DEFAULTS:
        return ""                                    # Requirement 6.5 / 3.4
    return "_cf{}_rho{}_src{}_mca{}".format(                                # Requirement 6.4
        _fmt_num(values[0]), _fmt_num(values[1]), values[2], _fmt_num(values[3]))

# Register the resolver with Hydra
OmegaConf.register_new_resolver("replace_slash", replace_slash)
OmegaConf.register_new_resolver("replace_substring", replace_substring)
OmegaConf.register_new_resolver("ccr_tag", ccr_tag)

if __name__ == "__main__":
    register_resolvers()

