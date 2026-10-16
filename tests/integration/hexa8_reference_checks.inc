// Generated from MYSTRAN/hexa_msc_comparison: expected values are MYSTRAN's Q1/E21 CHEXA8
// (MYSTRAN 19.0.0) results for these decks; comments give the reference and Simcenter values
// from MYSTRAN/hexa_msc_comparison/results.md.

    {"macneal_beams/CHEXA_straight_regular_extension", "sum", 0, {25, 26, 27, 28}, {0}, 0.0001185154, 5e-06, 0}, // sum of TX at grids 25,26,27,28; reference 0.00012, Simcenter 0.000118515
    {"macneal_beams/CHEXA_straight_regular_inplane_shear", "sum", 1, {25, 26, 27, 28}, {0}, 0.4239828, 5e-06, 0}, // sum of TY at grids 25,26,27,28; reference 0.4324, Simcenter 0.423983
    {"macneal_beams/CHEXA_straight_regular_outofplane_shear", "sum", 2, {25, 26, 27, 28}, {0}, 1.6949948, 5e-06, 0}, // sum of TZ at grids 25,26,27,28; reference 1.7284, Simcenter 1.695
    {"macneal_beams/CHEXA_straight_regular_twist", "diff", 1, {25, 27}, {0}, 0.002916396, 5e-06, 0}, // diff of TY at grids 25,27; reference 0.003208, Simcenter 0.0029164
    {"macneal_beams/CHEXA_straight_regular_twist", "diff", 2, {26, 25}, {0}, 0.005835542, 5e-06, 0}, // diff of TZ at grids 26,25; reference 0.006416, Simcenter 0.00583554
    {"macneal_beams/CHEXA_straight_trapezoidal_extension", "sum", 0, {25, 26, 27, 28}, {0}, 0.00011932008, 5e-06, 0}, // sum of TX at grids 25,26,27,28; reference 0.00012, Simcenter 0.000119286
    {"macneal_beams/CHEXA_straight_trapezoidal_inplane_shear", "sum", 1, {25, 26, 27, 28}, {0}, 0.020496488, 5e-06, 0}, // sum of TY at grids 25,26,27,28; reference 0.4324, Simcenter 0.0185392
    {"macneal_beams/CHEXA_straight_trapezoidal_outofplane_shear", "sum", 2, {25, 26, 27, 28}, {0}, 0.05377542, 5e-06, 0}, // sum of TZ at grids 25,26,27,28; reference 1.7284, Simcenter 0.0511751
    {"macneal_beams/CHEXA_straight_trapezoidal_twist", "diff", 1, {25, 27}, {0}, 0.002877176, 5e-06, 0}, // diff of TY at grids 25,27; reference 0.003208, Simcenter 0.0028303
    {"macneal_beams/CHEXA_straight_trapezoidal_twist", "diff", 2, {26, 25}, {0}, 0.005800634, 5e-06, 0}, // diff of TZ at grids 26,25; reference 0.006416, Simcenter 0.00570785
    {"macneal_beams/CHEXA_straight_parallelogram_extension", "sum", 0, {25, 26, 27, 28}, {0}, 0.00011931012, 5e-06, 0}, // sum of TX at grids 25,26,27,28; reference 0.00012, Simcenter 0.000119266
    {"macneal_beams/CHEXA_straight_parallelogram_inplane_shear", "sum", 1, {25, 26, 27, 28}, {0}, 0.2700115, 5e-06, 0}, // sum of TY at grids 25,26,27,28; reference 0.4324, Simcenter 0.116975
    {"macneal_beams/CHEXA_straight_parallelogram_outofplane_shear", "sum", 2, {25, 26, 27, 28}, {0}, 1.0140026, 5e-06, 0}, // sum of TZ at grids 25,26,27,28; reference 1.7284, Simcenter 0.994087
    {"macneal_beams/CHEXA_straight_parallelogram_twist", "diff", 1, {25, 27}, {0}, 0.002902464, 5e-06, 0}, // diff of TY at grids 25,27; reference 0.003208, Simcenter 0.00266369
    {"macneal_beams/CHEXA_straight_parallelogram_twist", "diff", 2, {26, 25}, {0}, 0.005829867, 5e-06, 0}, // diff of TZ at grids 26,25; reference 0.006416, Simcenter 0.00534904
    {"macneal_beams/CHEXA_curved_inplane_shear", "sum", 1, {25, 26, 27, 28}, {0}, 0.12878944, 5e-06, 0}, // sum of TY at grids 25,26,27,28; reference 0.34936, Simcenter 0.128788
    {"macneal_beams/CHEXA_curved_outofplane_shear", "sum", 2, {25, 26, 27, 28}, {0}, 1.6777786, 5e-06, 0}, // sum of TZ at grids 25,26,27,28; reference 2.0088, Simcenter 1.67777
    {"shell/CHEXA8_scordelis_lo", "sum", 2, {273, 562}, {0}, -0.6031264, 5e-06, 0}, // sum of TZ at grids 273,562; reference -0.6048, Simcenter -0.603129
    {"shell/CHEXA8_pinched_cylinder", "sum", 2, {1, 290}, {0}, -3.32831e-05, 5e-06, 0}, // sum of TZ at grids 1,290; reference -3.6496e-05, Simcenter -3.32833e-05
    {"shell/CHEXA8_pinched_hemisphere", "sum", 0, {1, 290}, {0}, 0.18516459, 5e-06, 0}, // sum of TX at grids 1,290; reference 0.188, Simcenter 0.185192
    {"shell/CHEXA8_twisted_beam_inplane", "sum", 2, {13, 26, 39, 52, 65, 78}, {0}, 0.032497878, 5e-06, 0}, // sum of TZ at grids 13,26,39,52,65,78; reference 0.032544, Simcenter 0.0323534
    {"shell/CHEXA8_twisted_beam_outofplane", "sum", 1, {13, 26, 39, 52, 65, 78}, {0}, 0.010425434, 5e-06, 0}, // sum of TY at grids 13,26,39,52,65,78; reference 0.010524, Simcenter 0.0103893
    {"shell/CHEXA8_plate_ss", "sum", 2, {1, 82}, {0}, -0.08881897, 5e-06, 0}, // sum of TZ at grids 1,82; reference -0.0887217, Simcenter -0.088819
    {"shell/CHEXA8_plate_clamped", "sum", 2, {1, 82}, {0}, -0.02763663, 5e-06, 0}, // sum of TZ at grids 1,82; reference -0.0276346, Simcenter -0.0276366
    {"modal/CHEXA8_plate_ss_modes", "freq", 0, {}, {1}, 48.28611, 5e-06, 0}, // mode 1 frequency (Hz); reference 47.9865
    {"modal/CHEXA8_plate_ss_modes", "freq", 0, {}, {2}, 123.1053, 5e-06, 0}, // mode 2 frequency (Hz); reference 119.966
    {"modal/CHEXA8_plate_ss_modes", "freq", 0, {}, {3}, 123.1054, 5e-06, 0}, // mode 3 frequency (Hz); reference 119.966
    {"modal/CHEXA8_plate_ss_modes", "freq", 0, {}, {4}, 197.0845, 5e-06, 0}, // mode 4 frequency (Hz); reference 191.946
    {"modal/CHEXA8_plate_ss_modes", "freq", 0, {}, {5}, 256.6488, 5e-06, 0}, // mode 5 frequency (Hz); reference 239.932
    {"modal/CHEXA8_plate_cantilever_modes", "freq", 0, {}, {1}, 8.446953, 5e-06, 0}, // mode 1 frequency (Hz); reference 8.48913
    {"modal/CHEXA8_plate_cantilever_modes", "freq", 0, {}, {2}, 20.71133, 5e-06, 0}, // mode 2 frequency (Hz); reference 20.7245
    {"modal/CHEXA8_plate_cantilever_modes", "freq", 0, {}, {3}, 52.54753, 5e-06, 0}, // mode 3 frequency (Hz); reference 52.0968
    {"modal/CHEXA8_plate_cantilever_modes", "freq", 0, {}, {4}, 66.78983, 5e-06, 0}, // mode 4 frequency (Hz); reference 66.4398
    {"modal/CHEXA8_plate_cantilever_modes", "freq", 0, {}, {5}, 76.19557, 5e-06, 0}, // mode 5 frequency (Hz); reference 75.6291
    {"modal/CHEXA8_beam_regular_modes", "freq", 0, {}, {1}, 1.434957, 5e-06, 0}, // mode 1 frequency (Hz); reference 1.41898
    {"modal/CHEXA8_beam_trapezoidal_modes", "freq", 0, {}, {1}, 7.277885, 5e-06, 0}, // mode 1 frequency (Hz); reference 1.41898
    {"modal/CHEXA8_beam_parallelogram_modes", "freq", 0, {}, {1}, 2.009678, 5e-06, 0}, // mode 1 frequency (Hz); reference 1.41898
    {"robustness/spec_8_cube_default_0.3", "zero_modes", 0, {}, {0}, 6, 0, 0}, // free-free element: exactly 6 rigid modes
    {"robustness/spec_8_cube_default_0.3", "eig", 0, {}, {7}, 4615385, 5e-06, 0}, // first deformation eigenvalue lambda_7
    {"robustness/spec_8_distorted_default_0.4999", "zero_modes", 0, {}, {0}, 6, 0, 0}, // free-free element: exactly 6 rigid modes
    {"robustness/spec_8_distorted_default_0.4999", "eig", 0, {}, {7}, 4040020, 5e-06, 0}, // first deformation eigenvalue lambda_7
    {"robustness/spec_8_needle_default_0.3", "zero_modes", 0, {}, {0}, 6, 0, 0}, // free-free element: exactly 6 rigid modes
    {"robustness/spec_8_needle_default_0.3", "eig", 0, {}, {7}, 11238.34, 5e-06, 0}, // first deformation eigenvalue lambda_7
    {"robustness/spec_8_trapezoid_IN3_0.3", "zero_modes", 0, {}, {0}, 6, 0, 0}, // free-free element: exactly 6 rigid modes
    {"robustness/spec_8_trapezoid_IN3_0.3", "eig", 0, {}, {7}, 3661018, 5e-06, 0}, // first deformation eigenvalue lambda_7
    {"robustness/spec_8_slab_default_0.4999", "zero_modes", 0, {}, {0}, 6, 0, 0}, // free-free element: exactly 6 rigid modes
    {"robustness/spec_8_slab_default_0.4999", "eig", 0, {}, {7}, 1539.134, 5e-06, 0}, // first deformation eigenvalue lambda_7
    {"robustness/cyl_8_0.3_default", "radial", 0, {1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66}, {0}, 1.88644587e-06, 5e-06, 0}, // mean u_r at r = a (Lame plane strain exact 1.90667e-06; ISOP FULL locks)
    {"robustness/cyl_8_0.4999_default", "radial", 0, {1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66}, {0}, 1.96781282e-06, 5e-06, 0}, // mean u_r at r = a (Lame plane strain exact 1.99997e-06; ISOP FULL locks)
    {"robustness/cyl_8_0.4999_FULL", "radial", 0, {1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66}, {0}, 1.2913071e-07, 5e-06, 0}, // mean u_r at r = a (Lame plane strain exact 1.99997e-06; ISOP FULL locks)
    {"robustness/aspect_8_default_1e+03", "along", 0, {3, 4, 7, 8}, {0.609668849699159, -0.530124078446138, -0.589298188660153}, 1e-12, 1e-06, 0}, // single 1000:1 element in tension, exact 1/(E ar^2)
    {"robustness/aspect_8_FULL_1e+20", "along", 0, {3, 4, 7, 8}, {0, 1, 0}, 1e-46, 1e-06, 0}, // single 1e+20:1 element in tension, exact 1/(E ar^2)
    {"robustness/inv_8_reference", "mean", 2, {5, 10, 15, 20, 25, 30, 35, 40, 45}, {0}, 0.000972150744, 5e-06, 0}, // mean tip TZ of the distorted 4x2x2 cantilever
    {"robustness/in8_blank", "mean", 2, {4, 8, 12, 16}, {0}, 0.001149976, 5e-06, 0}, // PSOLID IN blank and TWO select the same element
    {"robustness/in8_TWO", "mean", 2, {4, 8, 12, 16}, {0}, 0.001149976, 5e-06, 0}, // PSOLID IN blank and TWO select the same element
