import re
import matplotlib.pyplot as plt
import numpy as np

log_text = """
epoch 9 | step 10 | loss 1.475533
epoch 19 | step 20 | loss 0.731066
epoch 29 | step 30 | loss 0.529442
epoch 39 | step 40 | loss 0.539199
epoch 49 | step 50 | loss 0.375655
epoch 59 | step 60 | loss 0.388648
epoch 69 | step 70 | loss 0.168428
epoch 79 | step 80 | loss 0.199255
epoch 89 | step 90 | loss 0.166913
epoch 99 | step 100 | loss 0.170931
epoch 109 | step 110 | loss 0.114474
epoch 119 | step 120 | loss 0.136413
epoch 129 | step 130 | loss 0.117790
epoch 139 | step 140 | loss 0.138206
epoch 149 | step 150 | loss 0.191146
epoch 159 | step 160 | loss 0.258880
epoch 169 | step 170 | loss 0.058733
epoch 179 | step 180 | loss 0.063896
epoch 189 | step 190 | loss 0.056168
epoch 199 | step 200 | loss 0.079750
epoch 209 | step 210 | loss 0.059111
epoch 219 | step 220 | loss 0.072545
epoch 229 | step 230 | loss 0.085943
epoch 239 | step 240 | loss 0.117991
epoch 249 | step 250 | loss 0.229348
epoch 259 | step 260 | loss 0.240515
epoch 269 | step 270 | loss 0.057652
epoch 279 | step 280 | loss 0.045711
epoch 289 | step 290 | loss 0.072371
epoch 299 | step 300 | loss 0.043416
epoch 309 | step 310 | loss 0.035735
epoch 319 | step 320 | loss 0.118612
epoch 329 | step 330 | loss 0.047475
epoch 339 | step 340 | loss 0.076126
epoch 349 | step 350 | loss 0.182718
epoch 359 | step 360 | loss 0.048934
epoch 369 | step 370 | loss 0.043640
epoch 379 | step 380 | loss 0.046423
epoch 389 | step 390 | loss 0.058069
epoch 399 | step 400 | loss 0.040124
epoch 409 | step 410 | loss 0.032372
epoch 419 | step 420 | loss 0.092569
epoch 429 | step 430 | loss 0.066558
epoch 439 | step 440 | loss 0.032306
epoch 449 | step 450 | loss 0.106545
epoch 459 | step 460 | loss 0.040621
epoch 469 | step 470 | loss 0.032305
epoch 479 | step 480 | loss 0.048908
epoch 489 | step 490 | loss 0.028697
epoch 499 | step 500 | loss 0.020914
saved ./output/tm2m_finetune_new/step_0000500.pt
epoch 509 | step 510 | loss 0.105473
epoch 519 | step 520 | loss 0.034237
epoch 529 | step 530 | loss 0.025732
epoch 539 | step 540 | loss 0.177666
epoch 549 | step 550 | loss 0.042206
epoch 559 | step 560 | loss 0.336976
epoch 569 | step 570 | loss 0.042213
epoch 579 | step 580 | loss 0.058326
epoch 589 | step 590 | loss 0.114893
epoch 599 | step 600 | loss 0.030476
epoch 609 | step 610 | loss 0.041892
epoch 619 | step 620 | loss 0.033095
epoch 629 | step 630 | loss 0.028962
epoch 639 | step 640 | loss 0.700316
epoch 649 | step 650 | loss 0.034412
epoch 659 | step 660 | loss 0.028820
epoch 669 | step 670 | loss 0.024188
epoch 679 | step 680 | loss 0.025038
epoch 689 | step 690 | loss 0.031204
epoch 699 | step 700 | loss 0.033323
epoch 709 | step 710 | loss 0.029077
epoch 719 | step 720 | loss 0.022754
epoch 729 | step 730 | loss 0.020701
epoch 739 | step 740 | loss 0.019082
epoch 749 | step 750 | loss 0.023640
epoch 759 | step 760 | loss 0.056177
epoch 769 | step 770 | loss 0.026051
epoch 779 | step 780 | loss 0.151575
epoch 789 | step 790 | loss 0.040090
epoch 799 | step 800 | loss 0.024767
epoch 809 | step 810 | loss 0.018358
epoch 819 | step 820 | loss 0.026101
epoch 829 | step 830 | loss 0.031871
epoch 839 | step 840 | loss 0.028919
epoch 849 | step 850 | loss 0.018401
epoch 859 | step 860 | loss 0.067602
epoch 869 | step 870 | loss 0.017246
epoch 879 | step 880 | loss 0.018024
epoch 889 | step 890 | loss 1.024728
epoch 899 | step 900 | loss 0.025545
epoch 909 | step 910 | loss 0.037204
epoch 919 | step 920 | loss 0.084729
epoch 929 | step 930 | loss 0.039001
epoch 939 | step 940 | loss 0.027343
epoch 949 | step 950 | loss 0.016923
epoch 959 | step 960 | loss 1.264160
epoch 969 | step 970 | loss 0.045921
epoch 979 | step 980 | loss 0.095864
epoch 989 | step 990 | loss 0.021685
epoch 999 | step 1000 | loss 0.027502
saved ./output/tm2m_finetune_new/step_0001000.pt
epoch 1009 | step 1010 | loss 0.036274
epoch 1019 | step 1020 | loss 0.016330
epoch 1029 | step 1030 | loss 0.012693
epoch 1039 | step 1040 | loss 0.020741
epoch 1049 | step 1050 | loss 0.024938
epoch 1059 | step 1060 | loss 1.299699
epoch 1069 | step 1070 | loss 0.026672
epoch 1079 | step 1080 | loss 0.016376
epoch 1089 | step 1090 | loss 0.013047
epoch 1099 | step 1100 | loss 0.011575
epoch 1109 | step 1110 | loss 0.029168
epoch 1119 | step 1120 | loss 0.011011
epoch 1129 | step 1130 | loss 0.018443
epoch 1139 | step 1140 | loss 0.027274
epoch 1149 | step 1150 | loss 0.016212
epoch 1159 | step 1160 | loss 0.012520
epoch 1169 | step 1170 | loss 0.013942
epoch 1179 | step 1180 | loss 0.028285
epoch 1189 | step 1190 | loss 0.018570
epoch 1199 | step 1200 | loss 0.887719
epoch 1209 | step 1210 | loss 0.011718
epoch 1219 | step 1220 | loss 0.015907
epoch 1229 | step 1230 | loss 0.020239
epoch 1239 | step 1240 | loss 0.019909
epoch 1249 | step 1250 | loss 0.018460
epoch 1259 | step 1260 | loss 0.021643
epoch 1269 | step 1270 | loss 0.018262
epoch 1279 | step 1280 | loss 0.016216
epoch 1289 | step 1290 | loss 0.010357
epoch 1299 | step 1300 | loss 0.044220
epoch 1309 | step 1310 | loss 0.048018
epoch 1319 | step 1320 | loss 0.057252
epoch 1329 | step 1330 | loss 0.025252
epoch 1339 | step 1340 | loss 0.020726
epoch 1349 | step 1350 | loss 0.015119
epoch 1359 | step 1360 | loss 0.013225
epoch 1369 | step 1370 | loss 0.014518
epoch 1379 | step 1380 | loss 0.016828
epoch 1389 | step 1390 | loss 0.027391
epoch 1399 | step 1400 | loss 0.027284
epoch 1409 | step 1410 | loss 0.011522
epoch 1419 | step 1420 | loss 0.012790
epoch 1429 | step 1430 | loss 0.017271
epoch 1439 | step 1440 | loss 0.024412
epoch 1449 | step 1450 | loss 0.019598
epoch 1459 | step 1460 | loss 0.016497
epoch 1469 | step 1470 | loss 0.016845
epoch 1479 | step 1480 | loss 0.013266
epoch 1489 | step 1490 | loss 0.037393
epoch 1499 | step 1500 | loss 0.021392
saved ./output/tm2m_finetune_new/step_0001500.pt
epoch 1509 | step 1510 | loss 0.020710
epoch 1519 | step 1520 | loss 0.066475
epoch 1529 | step 1530 | loss 0.021215
epoch 1539 | step 1540 | loss 0.018901
epoch 1549 | step 1550 | loss 0.022674
epoch 1559 | step 1560 | loss 0.016515
epoch 1569 | step 1570 | loss 0.065959
epoch 1579 | step 1580 | loss 0.012238
epoch 1589 | step 1590 | loss 0.011565
epoch 1599 | step 1600 | loss 0.014780
epoch 1609 | step 1610 | loss 0.011603
epoch 1619 | step 1620 | loss 0.014299
epoch 1629 | step 1630 | loss 0.012272
epoch 1639 | step 1640 | loss 0.105344
epoch 1649 | step 1650 | loss 0.019640
epoch 1659 | step 1660 | loss 0.074330
epoch 1669 | step 1670 | loss 0.058741
epoch 1679 | step 1680 | loss 0.021256
epoch 1689 | step 1690 | loss 0.028249
epoch 1699 | step 1700 | loss 0.013885
epoch 1709 | step 1710 | loss 0.012130
epoch 1719 | step 1720 | loss 0.010550
epoch 1729 | step 1730 | loss 0.009425
epoch 1739 | step 1740 | loss 0.009527
epoch 1749 | step 1750 | loss 0.015669
epoch 1759 | step 1760 | loss 0.019897
epoch 1769 | step 1770 | loss 0.010397
epoch 1779 | step 1780 | loss 0.072169
epoch 1789 | step 1790 | loss 0.009761
epoch 1799 | step 1800 | loss 0.011587
epoch 1809 | step 1810 | loss 0.011879
epoch 1819 | step 1820 | loss 0.010428
epoch 1829 | step 1830 | loss 0.008151
epoch 1839 | step 1840 | loss 0.008721
epoch 1849 | step 1850 | loss 0.063946
epoch 1859 | step 1860 | loss 0.011036
epoch 1869 | step 1870 | loss 0.010692
epoch 1879 | step 1880 | loss 0.021455
epoch 1889 | step 1890 | loss 0.025510
epoch 1899 | step 1900 | loss 0.019071
epoch 1909 | step 1910 | loss 0.014051
epoch 1919 | step 1920 | loss 0.014714
epoch 1929 | step 1930 | loss 0.015411
epoch 1939 | step 1940 | loss 0.014149
epoch 1949 | step 1950 | loss 0.021435
epoch 1959 | step 1960 | loss 0.008837
epoch 1969 | step 1970 | loss 0.039324
epoch 1979 | step 1980 | loss 0.008308
epoch 1989 | step 1990 | loss 0.013725
epoch 1999 | step 2000 | loss 0.012372
saved ./output/tm2m_finetune_new/step_0002000.pt
epoch 2009 | step 2010 | loss 0.009128
epoch 2019 | step 2020 | loss 0.014526
epoch 2029 | step 2030 | loss 0.011265
epoch 2039 | step 2040 | loss 0.045736
epoch 2049 | step 2050 | loss 0.016806
epoch 2059 | step 2060 | loss 0.012081
epoch 2069 | step 2070 | loss 0.012737
epoch 2079 | step 2080 | loss 0.020928
epoch 2089 | step 2090 | loss 0.012811
epoch 2099 | step 2100 | loss 0.019520
epoch 2109 | step 2110 | loss 0.010320
epoch 2119 | step 2120 | loss 0.017164
epoch 2129 | step 2130 | loss 0.011735
epoch 2139 | step 2140 | loss 0.012060
epoch 2149 | step 2150 | loss 0.012205
epoch 2159 | step 2160 | loss 0.031141
epoch 2169 | step 2170 | loss 0.012433
epoch 2179 | step 2180 | loss 0.033961
epoch 2189 | step 2190 | loss 0.020435
epoch 2199 | step 2200 | loss 0.011724
epoch 2209 | step 2210 | loss 0.018826
epoch 2219 | step 2220 | loss 0.015175
epoch 2229 | step 2230 | loss 0.012101
epoch 2239 | step 2240 | loss 0.032512
epoch 2249 | step 2250 | loss 0.009297
epoch 2259 | step 2260 | loss 0.008916
epoch 2269 | step 2270 | loss 0.006615
epoch 2279 | step 2280 | loss 0.007576
epoch 2289 | step 2290 | loss 0.009071
epoch 2299 | step 2300 | loss 0.019758
epoch 2309 | step 2310 | loss 0.009193
epoch 2319 | step 2320 | loss 0.013942
epoch 2329 | step 2330 | loss 0.016496
epoch 2339 | step 2340 | loss 0.012186
epoch 2349 | step 2350 | loss 0.012633
epoch 2359 | step 2360 | loss 0.012677
epoch 2369 | step 2370 | loss 0.107113
epoch 2379 | step 2380 | loss 0.831820
epoch 2389 | step 2390 | loss 0.012792
epoch 2399 | step 2400 | loss 0.021735
epoch 2409 | step 2410 | loss 0.010333
epoch 2419 | step 2420 | loss 0.013152
epoch 2429 | step 2430 | loss 0.011474
epoch 2439 | step 2440 | loss 0.051923
epoch 2449 | step 2450 | loss 0.053223
epoch 2459 | step 2460 | loss 0.270816
epoch 2469 | step 2470 | loss 0.015251
epoch 2479 | step 2480 | loss 0.140246
epoch 2489 | step 2490 | loss 0.018319
epoch 2499 | step 2500 | loss 0.016349
saved ./output/tm2m_finetune_new/step_0002500.pt
epoch 2509 | step 2510 | loss 0.010181
epoch 2519 | step 2520 | loss 0.007938
epoch 2529 | step 2530 | loss 0.017504
epoch 2539 | step 2540 | loss 0.007387
epoch 2549 | step 2550 | loss 0.239993
epoch 2559 | step 2560 | loss 0.044225
epoch 2569 | step 2570 | loss 0.012069
epoch 2579 | step 2580 | loss 0.012007
epoch 2589 | step 2590 | loss 0.007607
epoch 2599 | step 2600 | loss 0.006039
epoch 2609 | step 2610 | loss 0.007450
epoch 2619 | step 2620 | loss 0.010871
epoch 2629 | step 2630 | loss 1.106950
epoch 2639 | step 2640 | loss 0.028274
epoch 2649 | step 2650 | loss 0.012081
epoch 2659 | step 2660 | loss 0.013572
epoch 2669 | step 2670 | loss 0.019045
epoch 2679 | step 2680 | loss 0.028648
epoch 2689 | step 2690 | loss 0.135258
epoch 2699 | step 2700 | loss 0.028050
epoch 2709 | step 2710 | loss 0.012228
epoch 2719 | step 2720 | loss 0.013162
epoch 2729 | step 2730 | loss 0.013281
epoch 2739 | step 2740 | loss 0.010524
epoch 2749 | step 2750 | loss 0.007731
epoch 2759 | step 2760 | loss 0.018347
epoch 2769 | step 2770 | loss 0.007914
epoch 2779 | step 2780 | loss 0.039846
epoch 2789 | step 2790 | loss 0.010091
epoch 2799 | step 2800 | loss 0.012424
epoch 2809 | step 2810 | loss 0.008410
epoch 2819 | step 2820 | loss 0.007157
epoch 2829 | step 2830 | loss 0.121081
epoch 2839 | step 2840 | loss 0.006436
epoch 2849 | step 2850 | loss 0.011029
epoch 2859 | step 2860 | loss 0.006989
epoch 2869 | step 2870 | loss 0.007730
epoch 2879 | step 2880 | loss 0.006327
epoch 2889 | step 2890 | loss 0.009784
epoch 2899 | step 2900 | loss 0.007059
epoch 2909 | step 2910 | loss 0.005372
epoch 2919 | step 2920 | loss 0.006163
epoch 2929 | step 2930 | loss 0.009248
epoch 2939 | step 2940 | loss 0.007117
epoch 2949 | step 2950 | loss 0.008921
epoch 2959 | step 2960 | loss 0.009643
epoch 2969 | step 2970 | loss 0.007474
epoch 2979 | step 2980 | loss 0.007209
epoch 2989 | step 2990 | loss 0.012373
epoch 2999 | step 3000 | loss 0.016426
saved ./output/tm2m_finetune_new/step_0003000.pt
epoch 3009 | step 3010 | loss 0.006943
epoch 3019 | step 3020 | loss 0.015337
epoch 3029 | step 3030 | loss 0.006407
epoch 3039 | step 3040 | loss 0.007854
epoch 3049 | step 3050 | loss 0.006199
epoch 3059 | step 3060 | loss 0.016083
epoch 3069 | step 3070 | loss 0.006885
epoch 3079 | step 3080 | loss 0.007731
epoch 3089 | step 3090 | loss 0.024436
epoch 3099 | step 3100 | loss 0.009197
epoch 3109 | step 3110 | loss 0.023148
epoch 3119 | step 3120 | loss 0.009940
epoch 3129 | step 3130 | loss 0.018926
epoch 3139 | step 3140 | loss 0.012933
epoch 3149 | step 3150 | loss 0.006895
epoch 3159 | step 3160 | loss 0.009152
epoch 3169 | step 3170 | loss 0.050406
epoch 3179 | step 3180 | loss 0.021500
epoch 3189 | step 3190 | loss 0.007186
epoch 3199 | step 3200 | loss 0.008309
epoch 3209 | step 3210 | loss 0.020073
epoch 3219 | step 3220 | loss 0.135604
epoch 3229 | step 3230 | loss 0.016240
epoch 3239 | step 3240 | loss 0.020561
epoch 3249 | step 3250 | loss 0.010580
epoch 3259 | step 3260 | loss 0.024354
epoch 3269 | step 3270 | loss 0.015496
epoch 3279 | step 3280 | loss 0.017673
epoch 3289 | step 3290 | loss 0.009179
epoch 3299 | step 3300 | loss 0.010626
epoch 3309 | step 3310 | loss 0.009970
epoch 3319 | step 3320 | loss 0.009828
epoch 3329 | step 3330 | loss 0.008152
epoch 3339 | step 3340 | loss 0.007747
epoch 3349 | step 3350 | loss 0.098600
epoch 3359 | step 3360 | loss 0.014557
epoch 3369 | step 3370 | loss 0.017186
epoch 3379 | step 3380 | loss 0.010330
epoch 3389 | step 3390 | loss 0.007414
epoch 3399 | step 3400 | loss 0.005844
epoch 3409 | step 3410 | loss 0.006109
epoch 3419 | step 3420 | loss 0.005097
epoch 3429 | step 3430 | loss 0.009396
epoch 3439 | step 3440 | loss 0.004222
epoch 3449 | step 3450 | loss 0.004358
epoch 3459 | step 3460 | loss 0.005187
epoch 3469 | step 3470 | loss 0.072090
epoch 3479 | step 3480 | loss 0.020911
epoch 3489 | step 3490 | loss 0.009580
epoch 3499 | step 3500 | loss 0.013048
saved ./output/tm2m_finetune_new/step_0003500.pt
epoch 3509 | step 3510 | loss 0.007832
epoch 3519 | step 3520 | loss 0.022052
epoch 3529 | step 3530 | loss 0.008569
epoch 3539 | step 3540 | loss 0.008051
epoch 3549 | step 3550 | loss 0.017892
epoch 3559 | step 3560 | loss 0.008711
epoch 3569 | step 3570 | loss 0.010256
epoch 3579 | step 3580 | loss 0.006513
epoch 3589 | step 3590 | loss 0.173356
epoch 3599 | step 3600 | loss 0.012952
epoch 3609 | step 3610 | loss 0.018525
epoch 3619 | step 3620 | loss 0.013431
epoch 3629 | step 3630 | loss 0.014534
epoch 3639 | step 3640 | loss 0.009355
epoch 3649 | step 3650 | loss 0.022856
epoch 3659 | step 3660 | loss 0.028045
epoch 3669 | step 3670 | loss 0.006769
epoch 3679 | step 3680 | loss 0.006352
epoch 3689 | step 3690 | loss 0.017421
epoch 3699 | step 3700 | loss 0.008422
epoch 3709 | step 3710 | loss 0.026384
epoch 3719 | step 3720 | loss 0.010058
epoch 3729 | step 3730 | loss 0.008492
epoch 3739 | step 3740 | loss 0.006705
epoch 3749 | step 3750 | loss 0.014481
epoch 3759 | step 3760 | loss 0.043639
epoch 3769 | step 3770 | loss 0.007842
epoch 3779 | step 3780 | loss 0.007252
epoch 3789 | step 3790 | loss 0.268310
epoch 3799 | step 3800 | loss 0.011169
epoch 3809 | step 3810 | loss 0.041199
epoch 3819 | step 3820 | loss 0.082510
epoch 3829 | step 3830 | loss 0.119898
epoch 3839 | step 3840 | loss 0.041741
epoch 3849 | step 3850 | loss 0.014901
epoch 3859 | step 3860 | loss 0.008206
epoch 3869 | step 3870 | loss 0.006205
epoch 3879 | step 3880 | loss 0.006401
epoch 3889 | step 3890 | loss 0.006570
epoch 3899 | step 3900 | loss 0.010605
epoch 3909 | step 3910 | loss 0.023091
epoch 3919 | step 3920 | loss 0.013619
epoch 3929 | step 3930 | loss 0.018132
epoch 3939 | step 3940 | loss 0.031425
epoch 3949 | step 3950 | loss 0.010206
epoch 3959 | step 3960 | loss 0.007468
epoch 3969 | step 3970 | loss 0.010227
epoch 3979 | step 3980 | loss 0.008988
epoch 3989 | step 3990 | loss 0.018251
epoch 3999 | step 4000 | loss 0.007257
saved ./output/tm2m_finetune_new/step_0004000.pt
epoch 4009 | step 4010 | loss 0.006274
epoch 4019 | step 4020 | loss 0.457897
epoch 4029 | step 4030 | loss 0.346740
epoch 4039 | step 4040 | loss 0.305810
epoch 4049 | step 4050 | loss 0.025595
epoch 4059 | step 4060 | loss 0.015630
epoch 4069 | step 4070 | loss 0.613678
epoch 4079 | step 4080 | loss 0.210670
epoch 4089 | step 4090 | loss 0.010399
epoch 4099 | step 4100 | loss 0.016433
epoch 4109 | step 4110 | loss 0.016186
epoch 4119 | step 4120 | loss 0.012606
epoch 4129 | step 4130 | loss 0.033072
epoch 4139 | step 4140 | loss 0.121862
epoch 4149 | step 4150 | loss 0.016538
epoch 4159 | step 4160 | loss 0.015164
epoch 4169 | step 4170 | loss 0.010852
epoch 4179 | step 4180 | loss 0.014950
epoch 4189 | step 4190 | loss 0.012495
epoch 4199 | step 4200 | loss 0.009414
epoch 4209 | step 4210 | loss 0.010382
epoch 4219 | step 4220 | loss 0.006838
epoch 4229 | step 4230 | loss 0.013184
epoch 4239 | step 4240 | loss 0.005567
epoch 4249 | step 4250 | loss 0.102335
epoch 4259 | step 4260 | loss 0.026606
epoch 4269 | step 4270 | loss 0.020795
epoch 4279 | step 4280 | loss 0.015268
epoch 4289 | step 4290 | loss 0.007034
epoch 4299 | step 4300 | loss 0.006044
epoch 4309 | step 4310 | loss 0.007421
epoch 4319 | step 4320 | loss 0.007337
epoch 4329 | step 4330 | loss 0.006762
epoch 4339 | step 4340 | loss 0.006544
epoch 4349 | step 4350 | loss 0.010010
epoch 4359 | step 4360 | loss 0.009592
epoch 4369 | step 4370 | loss 0.010208
epoch 4379 | step 4380 | loss 0.022530
epoch 4389 | step 4390 | loss 0.005923
epoch 4399 | step 4400 | loss 0.026447
epoch 4409 | step 4410 | loss 0.968707
epoch 4419 | step 4420 | loss 0.016815
epoch 4429 | step 4430 | loss 0.007156
epoch 4439 | step 4440 | loss 0.007379
epoch 4449 | step 4450 | loss 1.051968
epoch 4459 | step 4460 | loss 0.008082
epoch 4469 | step 4470 | loss 0.009969
epoch 4479 | step 4480 | loss 0.007275
epoch 4489 | step 4490 | loss 0.045569
epoch 4499 | step 4500 | loss 0.019512
saved ./output/tm2m_finetune_new/step_0004500.pt
epoch 4509 | step 4510 | loss 0.011005
epoch 4519 | step 4520 | loss 0.008576
epoch 4529 | step 4530 | loss 0.019670
epoch 4539 | step 4540 | loss 0.004928
epoch 4549 | step 4550 | loss 0.004313
epoch 4559 | step 4560 | loss 0.004785
epoch 4569 | step 4570 | loss 0.009313
epoch 4579 | step 4580 | loss 0.007979
epoch 4589 | step 4590 | loss 0.010268
epoch 4599 | step 4600 | loss 0.007451
epoch 4609 | step 4610 | loss 0.006500
epoch 4619 | step 4620 | loss 0.005117
epoch 4629 | step 4630 | loss 0.005066
epoch 4639 | step 4640 | loss 0.005112
epoch 4649 | step 4650 | loss 0.006199
epoch 4659 | step 4660 | loss 0.007642
epoch 4669 | step 4670 | loss 0.006024
epoch 4679 | step 4680 | loss 0.005664
epoch 4689 | step 4690 | loss 0.005862
epoch 4699 | step 4700 | loss 0.005168
epoch 4709 | step 4710 | loss 0.007073
epoch 4719 | step 4720 | loss 0.007169
epoch 4729 | step 4730 | loss 0.007352
epoch 4739 | step 4740 | loss 0.066553
epoch 4749 | step 4750 | loss 0.008394
epoch 4759 | step 4760 | loss 0.107242
epoch 4769 | step 4770 | loss 0.008725
epoch 4779 | step 4780 | loss 0.006401
epoch 4789 | step 4790 | loss 0.012196
epoch 4799 | step 4800 | loss 0.010760
epoch 4809 | step 4810 | loss 0.027690
epoch 4819 | step 4820 | loss 0.011743
epoch 4829 | step 4830 | loss 0.006947
epoch 4839 | step 4840 | loss 0.006418
epoch 4849 | step 4850 | loss 0.011565
epoch 4859 | step 4860 | loss 0.006852
epoch 4869 | step 4870 | loss 0.012275
epoch 4879 | step 4880 | loss 0.072231
epoch 4889 | step 4890 | loss 0.007741
epoch 4899 | step 4900 | loss 0.014400
epoch 4909 | step 4910 | loss 0.006957
epoch 4919 | step 4920 | loss 0.023908
epoch 4929 | step 4930 | loss 0.006089
epoch 4939 | step 4940 | loss 0.005140
epoch 4949 | step 4950 | loss 0.009806
epoch 4959 | step 4960 | loss 0.005397
epoch 4969 | step 4970 | loss 0.010324
epoch 4979 | step 4980 | loss 0.013485
epoch 4989 | step 4990 | loss 0.008493
epoch 4999 | step 5000 | loss 0.008310
"""

# ---- parse ----
steps, losses = [], []
pattern = re.compile(r"step (\d+) \| loss ([0-9.]+)")
for line in log_text.strip().splitlines():
    m = pattern.search(line)
    if m:
        steps.append(int(m.group(1)))
        losses.append(float(m.group(2)))

steps = np.array(steps)
losses = np.array(losses)

# ---- moving average ----
window = 10
if len(losses) >= window:
    kernel = np.ones(window) / window
    smooth = np.convolve(losses, kernel, mode="valid")
    smooth_steps = steps[window - 1:]
else:
    smooth, smooth_steps = losses, steps

# ---- plot ----
plt.figure(figsize=(10, 5))
plt.plot(steps, losses, marker="o", markersize=3, alpha=0.4, label="raw loss")
plt.plot(smooth_steps, smooth, color="tab:red", linewidth=2, label=f"moving avg (w={window})")

# mark checkpoint steps
ckpt_steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
for s in ckpt_steps:
    plt.axvline(s, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

plt.xlabel("step")
plt.ylabel("loss")
plt.title("TM2M fine-tuning loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
# plt.show()