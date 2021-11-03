"""
    @detector(expr)

An alternative to declaring the detector struct, clean! method and keyword constructor, direcly referring to
MLJModelInterface.@mlj_model.

Parameters
----------
    expr::Expr
An expression of a mutable struct defining a detector's hyperparameters.
"""
const var"@detector" = var"@mlj_model"

"""
    @default_frontend(detector)

Define a data front end for a given detector, which transforms the input data to `OutlierDetectionInterface.Data`.

Parameters
----------
    detector::T where T<:Detector
The detector datatype for which the data frontend should be defined.
"""
macro default_frontend(detector)
    detector = esc(detector)
    return quote
        MMI.reformat(::$detector, X::Data) = (X,)
        MMI.reformat(::$detector, X::Data, y) = (X, y)
        MMI.reformat(::$detector, X::Data, y, w) = (X, y, w) 
        MMI.reformat(::$detector, X) = (MMI.matrix(X, transpose=true),)
        MMI.reformat(::$detector, X, y) = (MMI.matrix(X, transpose=true), y)
        MMI.reformat(::$detector, X, y, w) = (MMI.matrix(X, transpose=true), y, w)
        MMI.selectrows(::$detector, I, Xmatrix) = (view(Xmatrix, ncolons(Xmatrix)..., I),)
        MMI.selectrows(::$detector, I, Xmatrix, y) = (view(Xmatrix, ncolons(Xmatrix)..., I), view(y, I))
        MMI.selectrows(::$detector, I, Xmatrix, y, w) = (view(Xmatrix, ncolons(Xmatrix)..., I), view(y, I), view(w, I))
    end
end

"""
    @default_metadata(detector,
                      uuid)

Define the default metadata for a given detector, which is useful when a detector is exported into MLJModels, such that
it can be directly loaded with MLJ. By default, we assume that a detector is exported on a package's top-level and we
set the `load_path` accordingly.

Additionally, we assume the following metadata defaults:

- `package_name` is equal to the `@__MODULE__`, where `@default_metadata` is used
- The detector is implemented in julia, `is_pure_julia=true`
- The detector is no wrapper, `is_wrapper=false`
- The package lives in the `OutlierDetectionJL` github organization

Parameters
----------
    detector::T where T<:Detector
The detector datatype for which the data frontend should be defined.

    uuid::String
The UUID of the detector's package.
"""
macro default_metadata(detector, uuid::String)
    detector = esc(detector)
    quote
        MMI.metadata_pkg($detector, package_name=string(@__MODULE__), package_uuid=$uuid,
                        package_url="https://github.com/OutlierDetectionJL/$(@__MODULE__).jl",
                        is_pure_julia=true, package_license="MIT", is_wrapper=false)
        MMI.load_path(::Type{$detector}) = string($detector)
    end
end
