
from dynaconf import Dynaconf, Validator
import validators

validators = [
    Validator("e4e_nas.url", required=True, cast=str, condition=validators.url),
    Validator("e4e_nas.username", required=True, cast=str),
    Validator("e4e_nas.password", required=True, cast=str),
    Validator(
        "fishsense_api.base_url", required=True, cast=str, condition=validators.url
    ),
    Validator("fishsense_api.username", required=True, cast=str),
    Validator("fishsense_api.password", required=True, cast=str),
]

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['settings.toml', '.secrets.toml'],
    validators=validators,
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
