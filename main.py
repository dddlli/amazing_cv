if __name__ == '__main__':
    import paddle
    from amazing_cv.attention import *
    from amazing_cv.mlp import *
    # input = paddle.randn([50, 49, 512])
    # a2 = DoubleAttention(512, 128, 128, True)
    # output = a2(input)
    # print(output.shape)

    # triplet = TripletAttention()
    # output = triplet(input)
    # print(output.shape)

    # eca = ECAAttention(kernel_size=3)
    # output = eca(input)
    # print(output.shape)

    # ssa = SimplifiedScaledDotProductAttention(d_model=512, h=8)
    # output = ssa(input, input, input)
    # print(output.shape)

    # input = paddle.randn([50, 512, 7, 7])
    # se = ShuffleAttention(channel=512, G=8)
    # output = se(input)
    # print(output.shape)

    # input = paddle.randn([50, 49, 512])
    # sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)
    # output = sa(input, input, input)
    # print(output.shape)

    # input = paddle.randn([50, 512, 7, 7])
    # sge = SpatialGroupEnhance(groups=8)
    # output = sge(input)
    # print(output.shape)

    # input = paddle.randn([50, 512, 7, 7])
    # se = SEAttention(channel=512, reduction=8)
    # output = se(input)
    # print(output.shape)

    # input = paddle.randn([50, 512, 7, 7])
    # psa = PSA(channel=512, reduction=8)
    # output = psa(input)
    # a = output.reshape([-1]).sum()
    # a.backward()
    # print(output.shape)

    # input = paddle.randn([50, 49, 512])
    # sa = MobileViTV2Attention(d_model=512)
    # output = sa(input)
    # print(output.shape)

    input = paddle.randn([50, 49, 512])
    sa = MUSEAttention(d_model=512, d_k=512, d_v=512, h=8)
    output = sa(input, input, input)
    print(output.shape)

    # num_tokens = 10000
    # bs = 50
    # len_sen = 49
    # num_layers = 6
    # input = paddle.randint(num_tokens, (bs, len_sen))  # bs,len_sen
    # gmlp = gMLP(num_tokens=num_tokens, len_sen=len_sen, dim=512, d_ff=1024)
    # output = gmlp(input)
    # print(output.shape)