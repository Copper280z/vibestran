// Device-resident CG state shared by the AMG shaders (std430; matches
// vulkan_amg.cpp's CgState). Every shader returns once flags are set, so
// iterations recorded past convergence do no work.
layout(set = 0, binding = STATE_BINDING) buffer State {
    double rho, alpha, beta, norm_b, target, rr;
    int flags, iterations, max_iterations, pad;
} st;
