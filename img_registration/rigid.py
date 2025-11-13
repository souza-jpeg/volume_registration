import itk
import SimpleITK as sitk
import numpy as np

def register_volumes(
    fixed_volume: np.ndarray,
    moving_volume: np.ndarray,
    sampling_percentage: float = 0.2,
    max_iterations: int = 200,
):
    """
    Aplica registro de imagem 3D entre dois volumes usando ITK (rigid + affine).

    Args:
        fixed_volume (np.ndarray): Volume de referência (resolução menor, ex: 64000nm)
        moving_volume (np.ndarray): Volume a ser alinhado (resolução maior, ex: 8000nm)
        sampling_percentage (float): Porcentagem de voxels amostrados no registro.
        max_iterations (int): Iterações máximas do otimizador.

    Returns:
        np.ndarray: Volume registrado (moving transformado para o espaço do fixed)
        itk.Transform: Transformação final estimada
    """

    # Converte para imagem ITK
    fixed = itk.image_from_array(fixed_volume.astype(np.float32))
    moving = itk.image_from_array(moving_volume.astype(np.float32))

    # Inicializa transformações (começa com transform rígida)
    initial_transform = itk.CenteredTransformInitializer.New(
        itk.Euler3DTransform[itk.F, 3],
        fixed,
        moving,
        itk.CenteredTransformInitializerFilter.GEOMETRY
    )

    transform = itk.Euler3DTransform[itk.F, 3].New()
    transform.SetParameters(initial_transform.GetParameters())

    # Define métrica (correlação mútua)
    metric = itk.MattesMutualInformationImageToImageMetricv4.New(FixedImage=fixed, MovingImage=moving)
    metric.SetNumberOfHistogramBins(50)

    # Otimizador (gradient descent)
    optimizer = itk.RegularStepGradientDescentOptimizerv4.New()
    optimizer.SetLearningRate(4.0)
    optimizer.SetMinimumStepLength(0.001)
    optimizer.SetRelaxationFactor(0.5)
    optimizer.SetNumberOfIterations(max_iterations)

    # Método de registro
    registration = itk.ImageRegistrationMethodv4.New(
        FixedImage=fixed,
        MovingImage=moving,
        Metric=metric,
        Optimizer=optimizer,
        Transform=transform
    )
    registration.SetMetricSamplingPercentage(sampling_percentage)
    registration.SetMetricSamplingStrategyToRandom()

    # Executa o registro
    registration.Update()
    final_transform = registration.GetTransform()

    # Aplica transformação no volume moving
    resampler = itk.ResampleImageFilter.New(
        Input=moving,
        Transform=final_transform,
        UseReferenceImage=True,
        ReferenceImage=fixed
    )
    resampler.Update()
    registered_volume = itk.array_from_image(resampler.GetOutput())

    return registered_volume, final_transform


def simple_register_volumes(
    fixed_volume: np.ndarray,
    moving_volume: np.ndarray,
    sampling_percentage: float = 0.2,
    max_iterations: int = 200,
):
    """
    Aplica registro de imagem 3D entre dois volumes usando SimpleITK (rigid + Mutual Information).

    Args:
        fixed_volume (np.ndarray): Volume de referência (resolução menor, ex: 64000nm)
        moving_volume (np.ndarray): Volume a ser alinhado (resolução maior, ex: 8000nm)
        sampling_percentage (float): Porcentagem de voxels amostrados no registro.
        max_iterations (int): Iterações máximas do otimizador.

    Returns:
        np.ndarray: Volume registrado (moving transformado para o espaço do fixed)
        sitk.Transform: Transformação final estimada
    """

    # --- 1️⃣ Converte arrays numpy para imagens SimpleITK ---
    fixed = sitk.GetImageFromArray(fixed_volume.astype(np.float32))
    moving = sitk.GetImageFromArray(moving_volume.astype(np.float32))

    # --- 2️⃣ Inicializa uma transformação rígida baseada na geometria ---
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # --- 3️⃣ Configura o método de registro ---
    registration = sitk.ImageRegistrationMethod()

    # Métrica: Mutual Information (boa para multimodal)
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    # Amostragem aleatória (para acelerar)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(sampling_percentage)

    # Interpolador linear (padrão)
    registration.SetInterpolator(sitk.sitkLinear)

    # Otimizador: Regular Step Gradient Descent
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=4.0,
        minStep=0.001,
        numberOfIterations=max_iterations,
        relaxationFactor=0.5,
    )

    registration.SetOptimizerScalesFromPhysicalShift()

    # Define a transformação inicial
    registration.SetInitialTransform(initial_transform, inPlace=False)

    # --- 4️⃣ Executa o registro ---
    final_transform = registration.Execute(fixed, moving)

    print(f"Valor final da métrica: {registration.GetMetricValue():.6f}")
    print(f"Convergiu: {registration.GetOptimizerConvergenceValue():.6f}")
    print(f"Nº de iterações: {registration.GetOptimizerIteration()}")

    # --- 5️⃣ Aplica a transformação ao volume moving ---
    registered = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID(),
    )

    registered_volume = sitk.GetArrayFromImage(registered)

    return registered_volume, final_transform


def register_images(fixed_image: np.ndarray,
                    moving_image: np.ndarray,
                    sampling_percentage: float = 0.2,
                    max_iterations: int = 200):
    """
    Aplica registro 2D rígido entre duas imagens usando SimpleITK.

    Args:
        fixed_image (np.ndarray): Imagem de referência (2D).
        moving_image (np.ndarray): Imagem a ser registrada (2D).
        sampling_percentage (float): Percentual de pixels amostrados na métrica.
        max_iterations (int): Iterações máximas do otimizador.

    Returns:
        np.ndarray: Imagem registrada (moving transformada para o espaço do fixed)
        sitk.Transform: Transformação final estimada
    """

    # Converte numpy arrays para imagens SimpleITK
    fixed = sitk.GetImageFromArray(fixed_image)
    moving = sitk.GetImageFromArray(moving_image)

    # Inicializa transformação rígida 2D (Euler)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # Configura o método de registro
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(sampling_percentage)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=4.0,
        minStep=0.001,
        numberOfIterations=max_iterations,
        relaxationFactor=0.5
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetInitialTransform(initial_transform, inPlace=False)

    # Executa registro
    final_transform = registration.Execute(fixed, moving)
    print("Métrica final:", registration.GetMetricValue())
    print("Número de iterações:", registration.GetOptimizerIteration())

    # Aplica a transformação na imagem moving
    registered = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID()
    )

    registered_array = sitk.GetArrayFromImage(registered)
    return registered_array, final_transform
