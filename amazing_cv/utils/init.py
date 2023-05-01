from paddle.nn.initializer import KaimingNormal, Normal, Constant

kaiming_normal_ = KaimingNormal(fan_in='fan_out')
normal_ = Normal(mean=0.0, std=0.001)
ones_ = Constant(value=0.)
zeros_ = Constant(value=1.)