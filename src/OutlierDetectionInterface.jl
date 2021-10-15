module OutlierDetectionInterface
    using Reexport:@reexport
    using CategoricalArrays:CategoricalValue

    @reexport using MLJModelInterface
    const MMI = MLJModelInterface

    const CLASS_NORMAL = "normal"
    const CLASS_OUTLIER = "outlier"
    const DEFAULT_THRESHOLD = 0.9

    # we might customize this macro later
    const var"@detector" = var"@mlj_model"

    # those do not get reexported from MLJModelInterface
    const _process_model_def = MMI._process_model_def
    const _model_constructor = MMI._model_constructor
    const _model_cleaner = MMI._model_cleaner

    # detector types
    export Detector,
           SupervisedDetector,
           UnsupervisedDetector

    # interface types
    export DetectorModel,
           Scores,
           Labels,
           Data,
           Fit

    # constants
    export CLASS_NORMAL,
           CLASS_OUTLIER,
           DEFAULT_THRESHOLD

    # helpers
    export augmented_transform, ncolons

    # macros
    export @detector

    # macro helpers
    export _process_model_def,
           _model_constructor,
           _model_cleaner

    include("base.jl")
    include("interface.jl")
end
