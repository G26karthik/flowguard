from flowguard.data import steady_targets_path


def test_targets_follow_the_discretisation():
    assert steady_targets_path("arith", "ghost") == "data/cache/2D_DarcyFlow_beta1.0_Train_steady.npy"
    assert steady_targets_path("harm", "face") == "data/cache/2D_DarcyFlow_beta1.0_Train_steady_harm_face.npy"
