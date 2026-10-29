// --+ options: stochastic,json=compute +--

var U2_Q_YED
    U2_G_YER
    U2_STN
    U2_EHIC
    U2_ESTN
    U2_G_EYER
    U2_HH_OCOR
    U2_HH_COR
    U2_H_Q_YER400
    U2_HH_COR1 ;

varexo res_U2_Q_YED
       res_U2_G_YER
       res_U2_STN
       res_U2_EHIC
       res_U2_ESTN
       res_U2_HH_OCOR
       res_U2_H_Q_YER
       res_ez ;

parameters u2_q_yed_ecm_u2_q_yed_L1
           u2_q_yed_ecm_u2_stn_L1
           u2_q_yed_u2_q_yed_L1
           u2_q_yed_u2_g_yer_L1
           u2_q_yed_u2_stn_L1
           u2_g_yer_ecm_u2_q_yed_L1
           u2_g_yer_ecm_u2_stn_L1
           u2_g_yer_u2_q_yed_L1
           u2_g_yer_u2_g_yer_L1
           u2_g_yer_u2_stn_L1
           u2_stn_ecm_u2_q_yed_L1
           u2_stn_ecm_u2_stn_L1
           u2_stn_u2_q_yed_L1
           u2_stn_u2_g_yer_L1
           u2_q_yed_ecm_u2_g_yer_L1
           u2_g_yer_ecm_u2_g_yer_L1
           u2_stn_ecm_u2_g_yer_L1
           u2_hh_ocor_ecm_u2_q_yed_L1
           u2_hh_ocor_ecm_u2_stn_L1
           u2_hh_ocor_ecm_u2_g_yer_L1
           u2_hh_ocor_u2_q_yed_L1
           u2_hh_ocor_u2_g_yer_L1
           u2_hh_ocor_u2_stn_L1
           u2_hh_ocor_ecm_u2_hh_ocor_L1
           u2_hh_ocor_u2_hh_ocor_L1
           beta
           ecm_pac
           u2_hh_cor_pac_u2_hh_cor_L1 ;


beta = 0.98 ;
ecm_pac = 0.2;
u2_hh_cor_pac_u2_hh_cor_L1 = 0.4;

model;

[name='U2_G_YER']
diff(U2_G_YER) =   u2_g_yer_ecm_u2_q_yed_L1 * (U2_Q_YED(-1) - U2_EHIC(-1))
                 + u2_g_yer_ecm_u2_stn_L1   * (U2_STN(-1)   - U2_ESTN(-1))
                 + u2_g_yer_ecm_u2_g_yer_L1 * (U2_G_YER(-1) - U2_G_EYER(-1))
                 + u2_g_yer_u2_q_yed_L1     * diff(U2_Q_YED(-1))
                 + u2_g_yer_u2_g_yer_L1     * diff(U2_G_YER(-1))
                 + u2_g_yer_u2_stn_L1       * diff(U2_STN(-1))
                 + res_U2_G_YER                                           ;


[name='U2_Q_YED']
diff(U2_Q_YED) =   u2_q_yed_ecm_u2_q_yed_L1 * (U2_Q_YED(-1) - U2_EHIC(-1))
                 + u2_q_yed_ecm_u2_stn_L1   * (U2_STN(-1)   - U2_ESTN(-1))
                 + u2_q_yed_ecm_u2_g_yer_L1 * (U2_G_YER(-1)- U2_G_EYER(-1))
                 + u2_q_yed_u2_q_yed_L1     * diff(U2_Q_YED(-1))
                 + u2_q_yed_u2_g_yer_L1     * diff(U2_G_YER(-1))
                 + u2_q_yed_u2_stn_L1       * diff(U2_STN(-1))
                 + res_U2_Q_YED                                           ;


[name='U2_ESTN']
U2_ESTN        =  U2_ESTN(-1) + res_U2_ESTN                               ;

[name='U2_EHIC']
U2_EHIC        =  U2_EHIC(-1) + res_U2_EHIC                               ;

[name='U2_G_EYER']
U2_G_EYER        =  0;

[name='U2_HH_OCOR']
diff(diff(log(U2_HH_OCOR))) =  u2_hh_ocor_ecm_u2_q_yed_L1   * (U2_Q_YED(-1) - U2_EHIC(-1))
                 + u2_hh_ocor_ecm_u2_stn_L1     * (U2_STN(-1)   - U2_ESTN(-1))
                 + u2_hh_ocor_ecm_u2_g_yer_L1   * (U2_G_YER(-1) - U2_G_EYER(-1))
                 + u2_hh_ocor_ecm_u2_hh_ocor_L1    * (diff(log(U2_HH_OCOR(-1))) - U2_H_Q_YER400(-1))
                 + u2_hh_ocor_u2_q_yed_L1       * diff(U2_Q_YED(-1))
                 + u2_hh_ocor_u2_g_yer_L1       * diff(U2_G_YER(-1))
                 + u2_hh_ocor_u2_stn_L1         * diff(U2_STN(-1))
                 + u2_hh_ocor_u2_hh_ocor_L1        * diff(diff(log(U2_HH_OCOR(-1))))
                 + res_U2_HH_OCOR                                        ;

[name='U2_H_Q_YER400']
U2_H_Q_YER400 = U2_H_Q_YER400(-1) + res_U2_H_Q_YER;

[name='U2_STN']
diff(U2_STN)   =   u2_stn_ecm_u2_q_yed_L1   * (U2_Q_YED(-1) - U2_EHIC(-1))
                 + u2_stn_ecm_u2_stn_L1     * (U2_STN(-1)   - U2_ESTN(-1))
                 + u2_stn_ecm_u2_g_yer_L1   * (U2_G_YER(-1) - U2_G_EYER(-1))
                 + u2_stn_u2_q_yed_L1       * diff(U2_Q_YED(-1))
                 + u2_stn_u2_g_yer_L1       * diff(U2_G_YER(-1))
                 + res_U2_STN                                             ;

[name='zpac']
diff(log(U2_HH_COR)) = ecm_pac*(log(U2_HH_OCOR(-1))-log(U2_HH_COR(-1))) +
                  u2_hh_cor_pac_u2_hh_cor_L1*diff(log(U2_HH_COR(-1))) +
                  pac_expectation(pacman)   +
                  res_ez;

[name='zpac1']
diff(log(U2_HH_COR1)) = ecm_pac*(log(U2_HH_OCOR(-1))-log(U2_HH_COR1(-1))) +
                  u2_hh_cor_pac_u2_hh_cor_L1*diff(log(U2_HH_COR1(-1))) +
                  pac_expectation(pacman1)   +
                  res_ez;
end;

// Calibration drawn by randn in Dynare's tcm6.mod (MATLAB's default stream).
u2_q_yed_ecm_u2_q_yed_L1 = 0.5376671395461;
u2_q_yed_ecm_u2_stn_L1 = 1.8338850145950865;
u2_q_yed_u2_q_yed_L1 = -2.258846861003648;
u2_q_yed_u2_g_yer_L1 = 0.8621733203681206;
u2_q_yed_u2_stn_L1 = 0.3187652398589808;
u2_g_yer_ecm_u2_q_yed_L1 = -1.3076882963052734;
u2_g_yer_ecm_u2_stn_L1 = -0.43359202230568356;
u2_g_yer_u2_q_yed_L1 = 0.3426244665386499;
u2_g_yer_u2_g_yer_L1 = 3.5783969397257605;
u2_g_yer_u2_stn_L1 = 2.769437029884877;
u2_stn_ecm_u2_q_yed_L1 = -1.3498869401565212;
u2_stn_ecm_u2_stn_L1 = 3.0349234663318545;
u2_stn_u2_q_yed_L1 = 0.7254042249461056;
u2_stn_u2_g_yer_L1 = -0.06305487318965619;
u2_q_yed_ecm_u2_g_yer_L1 = 0.7147429038260958;
u2_g_yer_ecm_u2_g_yer_L1 = -0.20496605829977463;
u2_stn_ecm_u2_g_yer_L1 = -0.12414434821631191;
u2_hh_ocor_ecm_u2_q_yed_L1 = 1.489697607785465;
u2_hh_ocor_ecm_u2_stn_L1 = 1.4090344898004792;
u2_hh_ocor_ecm_u2_g_yer_L1 = 1.4171924134296139;
u2_hh_ocor_u2_q_yed_L1 = 0.6714971336080805;
u2_hh_ocor_u2_g_yer_L1 = -1.2074869226850378;
u2_hh_ocor_u2_stn_L1 = 0.7172386513288385;
u2_hh_ocor_ecm_u2_hh_ocor_L1 = 1.6302352891647292;
u2_hh_ocor_u2_hh_ocor_L1 = 0.4888937703117894;

trend_component_model(model_name=toto, eqtags=['U2_Q_YED', 'U2_G_YER', 'U2_STN', 'U2_EHIC', 'U2_ESTN', 'U2_HH_OCOR', 'U2_H_Q_YER400'], targets=['U2_EHIC', 'U2_ESTN', 'U2_H_Q_YER400']);
trend_component_model(model_name=titi, eqtags=['U2_Q_YED', 'U2_G_YER', 'U2_STN', 'U2_EHIC', 'U2_ESTN', 'U2_HH_OCOR', 'U2_H_Q_YER400'], targets=['U2_H_Q_YER400', 'U2_ESTN', 'U2_EHIC']);
pac_model(auxiliary_model_name=toto, discount=beta, model_name=pacman, growth = U2_H_Q_YER400);
pac.initialize('pacman');
pac_model(auxiliary_model_name=titi, discount=beta, model_name=pacman1, growth = U2_H_Q_YER400);
pac.initialize('pacman1');
