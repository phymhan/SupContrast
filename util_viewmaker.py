import viewmaker


def create_viewmaker(config=None):
    view_model = viewmaker.Viewmaker(
        num_channels=3,
        distortion_budget=0.05,
        activation='relu',
        clamp=True,
        frequency_domain=False,
        downsample_to=False,
        num_res_blocks=3 or 5,
    )
    return view_model
