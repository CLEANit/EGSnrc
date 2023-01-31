      program examin
      real*8 GE,GMFP,RAYCOR,GBR1,GBR2,COMPT
      real*8 PHOTOE,TOTICM,TOTGCM
      real*8 EIE,EFRACT,TMXS,DEDXE
      real*8 SIGE,SIGP,BREME,BREMP
      real*8 DY(300)
      real*8 HISTXMIN
      character*512 toUpper
      CHARACTER*60 TITLE,XAXIS,YAXISPcom,YAXISPmfp,YAXISE,OUTFILE,YAXISE
     *mfp, SUBTITLE,SERIES,CURVE_LABELS(10)
      CHARACTER*256 FILENAME
      CHARACTER*256 the_title
      CHARACTER*24 material
      CHARACTER*2 CURVE_SPECS(10)
      CHARACTER*100 SCRIPT(3)
      CHARACTER*50 examin_pxsections
      CHARACTER*50 compton_xsections
      INTEGER NPOINTS(10),EPOPT,PLTOPT,NCURVE
      INTEGER IOPT,INDIV,IFILE,IPLOTE,IPLOTG,NUNIT
      INTEGER IDO
      INTEGER IFLAG1,IFLAG2,I1,I2,ICURVE
      INTEGER IFIRSTE
      INTEGER FILELENGTH
      integer egs_system,istat, p_comp
      real CPUT0,CPUT1,CPUT2,TDUM
      real egs_tot_time, egs_etime
      COMMON/EPCONT/EDEP,EDEP_LOCAL,TSTEP,TUSTEP,USTEP,TVSTEP,VSTEP, RHO
     *F,EOLD,ENEW,EKE,ELKE,GLE,E_RANGE, x_final,y_final,z_final, u_final
     *,v_final,w_final, IDISC,IROLD,IRNEW,IAUSFL(35)
      DOUBLE PRECISION EDEP,  EDEP_LOCAL
      real*8 TSTEP,  TUSTEP,  USTEP,  VSTEP,  TVSTEP,  RHOF,  EOLD,  ENE
     *W,  EKE,  ELKE,  GLE,  E_RANGE, x_final,y_final,z_final,  u_final,
     *v_final,w_final
      integer*4 IDISC,  IROLD,  IRNEW,  IAUSFL
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/BOUNDS/ECUT(1),PCUT(1),VACDST
      real*8 ECUT,  PCUT,  VACDST
      COMMON/MISC/  DUNIT,KMPI,KMPO,RHOR(1),MED(1),IRAYLR(1),IPHOTONUCR(
     *1)
      real*8 DUNIT,  RHOR
      integer*4 KMPI,  KMPO
      integer*2 MED,  IRAYLR,  IPHOTONUCR
      COMMON/ELECIN/ esig_e(1),psig_e(1), esige_max, psige_max, range_ep
     *(0:1,500,1), E_array(500,1), etae_ms0(500,1),etae_ms1(500,1),etap_
     *ms0(500,1),etap_ms1(500,1),q1ce_ms0(500,1),q1ce_ms1(500,1),q1cp_ms
     *0(500,1),q1cp_ms1(500,1),q2ce_ms0(500,1),q2ce_ms1(500,1),q2cp_ms0(
     *500,1),q2cp_ms1(500,1),blcce0(500,1),blcce1(500,1), EKE0(1),EKE1(1
     *), XR0(1),TEFF0(1),BLCC(1),XCC(1), ESIG0(500,1),ESIG1(500,1),PSIG0
     *(500,1),PSIG1(500,1),EDEDX0(500,1),EDEDX1(500,1),PDEDX0(500,1),PDE
     *DX1(500,1),EBR10(500,1),EBR11(500,1),PBR10(500,1),PBR11(500,1),PBR
     *20(500,1),PBR21(500,1),TMXS0(500,1),TMXS1(500,1), expeke1(1), IUNR
     *ST(1),EPSTFL(1),IAPRIM(1), sig_ismonotone(0:1,1)
      real*8 esig_e,   psig_e,   esige_max,  psige_max,  range_ep,  E_ar
     *ray,  etae_ms0,etae_ms1,  etap_ms0,etap_ms1,  q1ce_ms0,q1ce_ms1,
     *q1cp_ms0,q1cp_ms1,  q2ce_ms0,q2ce_ms1,  q2cp_ms0,q2cp_ms1,  blcce0
     *,blcce1,   expeke1,  EKE0,EKE1, XR0,  TEFF0,  BLCC,  XCC,  ESIG0,E
     *SIG1,  PSIG0,PSIG1,  EDEDX0,EDEDX1,  PDEDX0,PDEDX1,  EBR10,EBR11,
     * PBR10,PBR11,  PBR20,PBR21,  TMXS0,TMXS1
      integer*4 IUNRST,  EPSTFL,  IAPRIM
      logical sig_ismonotone
      COMMON/PHOTIN/ EBINDA(1), GE0(1),GE1(1), GMFP0(2000,1),GMFP1(2000,
     *1),GBR10(2000,1),GBR11(2000,1),GBR20(2000,1),GBR21(2000,1), RCO0(1
     *),RCO1(1), RSCT0(100,1),RSCT1(100,1), COHE0(2000,1),COHE1(2000,1),
     *  PHOTONUC0(2000,1),PHOTONUC1(2000,1), DPMFP, MPGEM(1,1), NGR(1)
      real*8 EBINDA,  GE0,GE1,  GMFP0,GMFP1,  GBR10,GBR11,  GBR20,GBR21,
     *  RCO0,RCO1,  RSCT0,RSCT1,  COHE0,COHE1,   PHOTONUC0,PHOTONUC1,  D
     *PMFP
      integer*4
     *                  MPGEM,
     *                          NGR
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common/compton_data/ iz_array(1538),  be_array(1538),  Jo_array(15
     *38),  erfJo_array(1538),   ne_array(1538),  shn_array(1538),
     *shell_array(200,1), eno_array(200,1), eno_atbin_array(200,1), n_sh
     *ell(1), radc_flag,  ibcmp(1)
      integer*4 iz_array,ne_array,shn_array,eno_atbin_array, shell_array
     *,n_shell,radc_flag
      real*8 be_array,Jo_array,erfJo_array,eno_array
      integer*2 ibcmp
      COMMON/EDGE/binding_energies(30,100), interaction_prob(6,100), rel
     *axation_prob(39,100), edge_energies(16,100), edge_number(100), edg
     *e_a(16,100), edge_b(16,100), edge_c(16,100), edge_d(16,100), IEDGF
     *L(1),IPHTER(1)
      real*8 binding_energies,  interaction_prob,    relaxation_prob,  e
     *dge_energies,  edge_a,edge_b,edge_c,edge_d
      integer*2 IEDGFL,  IPHTER
      integer*4 edge_number
      common/ET_control/ smaxir(1),estepe,ximax,  skindepth_for_bca,tran
     *sport_algorithm, bca_algorithm,exact_bca,spin_effects
      real*8 smaxir,  estepe,  ximax,      skindepth_for_bca
      integer*4 transport_algorithm, bca_algorithm
      logical exact_bca,  spin_effects
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      common/x_options/eadl_relax,  mcdf_pe_xsections
      logical eadl_relax, mcdf_pe_xsections
      real*8 ETAB(16),PLOTE(300),PLOTEM(300), PLOTEEN(300), PLOTG(300),P
     *LOTGEN(300), PLOTEMP(300), PLOTEMS(300), PHOTE_COMP(300), COMP_COM
     *P(300),PAIR_COMP(300), COHER_COMP(300)
      DATA ETAB/1.,1.25,1.5,1.75,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.,7.,8.,9.
     */
      call egs_init
      write(i_log,*) 'Media available in this data file are: '
      istat = egs_system('cat '//pegs_file(:lnblnk1(pegs_file))//' | gre
     *p MED')
      YAXISPcom = 'fractional component of photon cross section'
      YAXISPmfp = 'mean free path / cm'
      YAXISEmfp = 'mean free path / cm'
      YAXISE = 'dE/drhoX MeV/g/cm\\S2\\N'
      XAXIS = 'kinetic energy / MeV'
      HISTXMIN = 0.0
      INTERACTIVE = 1
      IF ((INTERACTIVE .EQ. 1)) THEN
        SCRIPT(1)='#!/bin/csh'
        SCRIPT(2)='#This is xmgr_script used with examin'
      END IF
      ibcmp(1) =0
      iedgfl(1) = 0
      eadl_relax = .false.
      spin_effects = .false.
      mcdf_pe_xsections = .false.
      xsec_out = 0
      DO 1011 I=1,300
        DY(I) = 0.0
1011  CONTINUE
1012  CONTINUE
      WRITE(6,1020)
1020  FORMAT(' Material identifier: ',$)
      READ(5,1030)(MEDIA(J,1),J=1,24)
1030  FORMAT(24A1)
      DO 1041 J=1,24
        material(J:J)=MEDIA(J,1)
1041  CONTINUE
1042  CONTINUE
      FILENAME(1:20)=material(1:20)
      WRITE(6,1050)FILENAME(1:20)
1050  FORMAT(///' Title after  ', A20, $)
      READ(5,1060)FILENAME(21:256)
1060  FORMAT(A256)
      the_title=FILENAME(21:256)
      WRITE(6,1070)
1070  FORMAT(' Electrons and photons(0), only electrons(1) or photons(2)
     *: ',$)
      READ(5,1080)IOPT
1080  FORMAT(I3)
      IF ((IOPT.EQ.0 .OR. IOPT .EQ. 2)) THEN
        write(*,'(a)') ' Select photon cross section data base! '
        write(*,'(a,a)')' ENTER defaults to: ',photon_xsections(:lnblnk1
     *  (photon_xsections))
        write(*,'(a$)') ' Photon xsections [si,xcom,mcdf-xcom,epdl,mcdf-
     *epdl,pegs4,user-defined]:'
        read(*,'(a)') examin_pxsections
        IF ((examin_pxsections(:lnblnk1(examin_pxsections)).ne.char(32))
     *  ) THEN
          photon_xsections = examin_pxsections(:lnblnk1(examin_pxsection
     *    s))
        END IF
        IF (( toUpper( examin_pxsections(:lnblnk1(examin_pxsections)) )
     *  .EQ. 'MCDF-XCOM' )) THEN
          mcdf_pe_xsections = .true.
          photon_xsections = 'xcom'
        ELSE IF(( toUpper( examin_pxsections(:lnblnk1(examin_pxsections)
     *  ) ) .EQ. 'MCDF-EPDL' )) THEN
          mcdf_pe_xsections = .true.
          photon_xsections = 'epdl'
        ELSE
          mcdf_pe_xsections = .false.
        END IF
        xsec_out = 1
        WRITE(6,1090)
1090    FORMAT(' Include Rayleigh(coherent)scattering(1) or not(0): ',$)
        READ(5,1100)IRAYLR(1)
1100    FORMAT(I10)
        write(*,'(3a$)') ' Compton xsections (default, xcom, user-define
     *d) [', comp_xsections(:lnblnk1(comp_xsections)),']:'
        read(*,'(a)') compton_xsections
        IF ((compton_xsections(:lnblnk1(compton_xsections)).ne.char(32))
     *  ) THEN
          comp_xsections = compton_xsections(:lnblnk1(compton_xsections)
     *    )
        END IF
        IF (( comp_xsections(:lnblnk1(comp_xsections)) .NE. 'default'))
     *  THEN
          ibcmp(1)=3
        ELSE
          WRITE(6,1110)
1110      FORMAT(' Include bound Compton(1) or not(0): ',$)
          READ(5,1120)IBCMP(1)
1120      FORMAT(I10)
          IF ((IBCMP(1).EQ.1)) THEN
            IBCMP(1)=3
          END IF
        END IF
        IF ((IBCMP(1) .GT. 0)) THEN
          iedgfl(1) = 1
          WRITE(6,1130)
1130      FORMAT(' init_compton: turning relaxations on')
        END IF
        WRITE(6,1140)
1140    FORMAT(' Include radiative Compton corrections(1) or not(0): ',$
     *)
        READ(5,1150)RADC_FLAG
1150    FORMAT(I10)
        WRITE(6,1160)
1160    FORMAT(' Print out absolute(0) or relative(1) components: ',$)
        READ(5,1170)p_comp
1170    FORMAT(I10)
      ELSE
        IRAYLR(1) = 0
        IBCMP(1) = 0
        RADC_FLAG = 0
      END IF
      WRITE(6,1180)
1180  FORMAT(' Tables(0) or individual energies(1): ',$)
      READ(5,1190)INDIV
1190  FORMAT(I10)
      WRITE(6,1200)
1200  FORMAT(' Output  terminal(0),  disk file(1), terminal&plot(2), dis
     *k&plot(3): ',$)
      READ(5,1210)IFILE
1210  FORMAT(I10)
      IF ((IFILE.LT.0 .OR. IFILE.GT.4)) THEN
        IFILE=0
      END IF
      IPLOTE=0
      IPLOTG=0
      IF ((IFILE .EQ. 0 .OR. IFILE .EQ. 2)) THEN
        NUNIT=6
      ELSE
        NUNIT=1
      END IF
      WRITE(6,1220)
1220  FORMAT(/' Call HATCH')
      CALL HATCH
      WRITE(6,1230)
1230  FORMAT(' Call to HATCH completed'/)
      NMED=1
      MED(1)=1
      MEDIUM=MED(1)
      DUNIT=1.0
      IRL=1
      IF ((IOPT .NE. 1)) THEN
        IF ((AP(1).LT.UP(1))) THEN
1240      WRITE(NUNIT,1250) (MEDIA(J,1),J=1,10),the_title(:lnblnk1(the_t
     *    itle))
1250      FORMAT('Summary of photon data: MEDIUM = ',10A1, ' Title= ',A/
     *T60,$)
          call egs_fdate(nunit)
          write(nunit,1260)
1260      format(/' ',79('='))
          WRITE(NUNIT,1270) AE(1)-PRM,UE(1)-PRM,AP(1),UP(1)
1270      FORMAT(/' Electron kinetic energy range:',T35,2F12.3,' MeV'/ '
     * Photon energy range:          ',T35,2F12.3,' MeV')
          WRITE(NUNIT,1280) RLC(MEDIUM),RHO(MEDIUM)
1280      FORMAT(/' Radiation length = ',T35,F13.6,' cm'/ ' Density = ',
     *T35,F13.6,' g/cm**3')
          WRITE(NUNIT,1290) photon_xsections(:lnblnk1(photon_xsections))
1290      FORMAT( ' Photon data set: ', A,$)
          IF ((p_comp .EQ. 0)) THEN
            write(nunit,'(a)') ' (absolute components)'
          ELSE
            write(nunit,'(a)') ' (relative components)'
          END IF
          IF ((mcdf_pe_xsections)) THEN
            write(nunit,'(a)') ' Renormalized photoelectric cross sectio
     *ns by Sabbatucci and Salvat'
          END IF
          IF (( comp_xsections(:lnblnk1(comp_xsections)) .NE. 'default')
     *    ) THEN
            write(NUNIT,'(2a)') ' Incoherent scatter data: ', comp_xsect
     *      ions(:lnblnk1(comp_xsections))
          ELSE
            write(NUNIT,'(a$)') ' Using EGSnrc incoherent scatter approa
     *ch: '
            IF ((IBCMP(1).GT.1)) THEN
              write(NUNIT,*) 'Klein-Nishina with binding effects include
     *d'
            ELSE
              write(NUNIT,*) 'Klein-Nishina without binding corrections'
            END IF
          END IF
          IF ((xsec_out .EQ. 1 )) THEN
            WRITE(NUNIT,1300)
1300        FORMAT( ' Will print .xsections file')
          ELSE
            WRITE(NUNIT,1310)
1310        FORMAT(' Will not print .xsections file')
          END IF
          IF ((IRAYLR(1) .EQ. 0)) THEN
            WRITE(NUNIT,1320)
1320        FORMAT(//4x,'Photon En',9x,'GMFP',8x,'Photoelec', 7x,'Compto
     *n',10x,'Pair',17x,'Total')
            IF ((p_comp .EQ. 1)) THEN
              WRITE(NUNIT,1330)
1330          FORMAT(21x,'(cm)',54x,'cm**2/g',8x,'cm**-1')
            ELSE
              WRITE(NUNIT,1340)
1340          FORMAT(21x,'(cm)',4(9x,'cm**2/g'),4x,'cm**2/g',8x,'cm**-1'
     *)
            END IF
          ELSE
            WRITE(NUNIT,1350)
1350        FORMAT(//4x,'Photon En',9x,'GMFP ',8x,'Rayleigh' ,7x,'Photoe
     *lec',7x,'Compton',9x,'Pair',17x,'Total')
            IF ((p_comp .EQ. 1)) THEN
              WRITE(NUNIT,1360)
1360          FORMAT(21x,'(cm)',67x,'cm**2/g',8x,'cm**-1')
            ELSE
              WRITE(NUNIT,1340)
1370          FORMAT(21x,'(cm)',4(9x,'cm**2/g'),4x,'cm**2/g',8x,'cm**-1'
     *)
            END IF
          END IF
          IFLAG1=0
          IFLAG2=0
          DO 1381 I1=1,8
            DO 1391 I2=1,16
1400          IF ((INDIV .EQ. 1)) THEN
                WRITE(6,1410)
1410            FORMAT(' Photon energy (MeV, 0=>go to electrons): ',$)
                READ(5,'(F12.0)',ERR=1400) GE
                IF ((GE .GT. UP(1) .OR. GE .LT. AP(1))) THEN
                  WRITE(6,1420)AP(1),UP(1)
1420              FORMAT(T20,'   must be in range',2(1Pe10.3))
                END IF
                IF ((GE .LE. 0.0)) THEN
                  GO TO 1430
                END IF
              ELSE
                GE=ETAB(I2)*10.**(I1-4)
              END IF
              IF ((GE .LE. AP(1))) THEN
                IF ((IFLAG1 .EQ. 0)) THEN
                  IFLAG1=1
                  GE=AP(1)
                ELSE
                  GE=0.0
                END IF
              END IF
              IF ((GE .GE. UP(1))) THEN
                IF ((IFLAG2 .EQ. 0)) THEN
                  IFLAG2=1
                  GE = UP(1)
                ELSE
                  GE=1.E30
                END IF
              END IF
              IF ((GE .GE. AP(1)-0.00001 .AND. GE .LE. UP(1)+0.001)) THE
     *        N
                GLE=LOG(GE)
                LGLE=GE1(MEDIUM)*GLE+GE0(MEDIUM)
                IF ((IRAYLR(1) .EQ. 1)) THEN
                  GMFP=1.0
                  IF ((IRAYLR(IRL).EQ.1)) THEN
                    COHFAC=COHE1(LGLE,MEDIUM)*GLE+COHE0(LGLE,MEDIUM)
                    GMFP=GMFP*COHFAC
                  END IF
                  RAYCOR=GMFP
                END IF
                GMFP=GMFP1(LGLE,MEDIUM)*GLE+GMFP0(LGLE,MEDIUM)
                IF ((IRAYLR(IRL).EQ.1)) THEN
                  COHFAC=COHE1(LGLE,MEDIUM)*GLE+COHE0(LGLE,MEDIUM)
                  GMFP=GMFP*COHFAC
                END IF
                GBR1=GBR11(LGLE,MEDIUM)*GLE+GBR10(LGLE,MEDIUM)
                GBR2=GBR21(LGLE,MEDIUM)*GLE+GBR20(LGLE,MEDIUM)
                COMPT = MAX(GBR2-GBR1, 0.0)
                PHOTOE= MAX(1.-GBR2, 0.0)
                TOTICM=1./GMFP
                TOTGCM=TOTICM/RHO(1)
                IF ((I2 .EQ. 1)) THEN
                  WRITE(NUNIT,1440)
                END IF
1440            FORMAT('   ')
                IF ((IRAYLR(1) .EQ. 0)) THEN
                  IF ((p_comp .EQ. 0)) THEN
                    WRITE(NUNIT,1450) GE,GMFP,PHOTOE*TOTGCM,COMPT*TOTGCM
     *               ,GBR1*TOTGCM,TOTGCM,TOTICM
1450                FORMAT(7(1PE15.6))
                  ELSE
                    WRITE(NUNIT,1460) GE,GMFP,PHOTOE,COMPT ,GBR1,TOTGCM,
     *              TOTICM
1460                FORMAT(7(1PE15.6))
                  END IF
                ELSE
                  IF ((p_comp .EQ. 0)) THEN
                    WRITE(NUNIT,1470) GE,GMFP,(1.-RAYCOR)*TOTGCM ,PHOTOE
     *              *RAYCOR*TOTGCM,COMPT*RAYCOR*TOTGCM,GBR1*RAYCOR*TOTGC
     *              M ,TOTGCM,TOTICM
1470                FORMAT(8(1PE15.6))
                  ELSE
                    WRITE(NUNIT,1480) GE,GMFP,(1.-RAYCOR) ,PHOTOE*RAYCOR
     *              ,COMPT*RAYCOR,GBR1*RAYCOR ,TOTGCM,TOTICM
1480                FORMAT(8(1PE15.6))
                  END IF
                END IF
                IF ((IFILE .GE. 2)) THEN
                  IPLOTG=IPLOTG+1
                  IF ((IRAYLR(1).EQ.0)) THEN
                    RAYCOR = 1.00
                  END IF
                  PLOTGEN(IPLOTG)=GE
                  PLOTG(IPLOTG)=GMFP
                  PHOTE_COMP(IPLOTG) = PHOTOE*RAYCOR
                  COMP_COMP(IPLOTG) = COMPT*RAYCOR
                  PAIR_COMP(IPLOTG) = GBR1*RAYCOR
                  COHER_COMP(IPLOTG) = 1.0 - RAYCOR
                END IF
              END IF
1391        CONTINUE
1392        CONTINUE
1381      CONTINUE
1382      CONTINUE
        END IF
      END IF
1430  IF ((IOPT .NE. 2)) THEN
        WRITE(NUNIT,1490) (MEDIA(J,1),J=1,10),the_title(:lnblnk1(the_tit
     *  le))
1490    FORMAT(////'Summary electron data: MEDIUM=',10A1, ' Title:',A30/
     *T60,$)
        call egs_fdate(nunit)
        write(nunit,1500)
1500    format(/' ',79('='))
        WRITE(NUNIT,1510) AE(1)-PRM,UE(1)-PRM,AP(1),UP(1)
1510    FORMAT(/' Electron kinetic energy range:',T35,2F12.3,' MeV'/ ' P
     *hoton energy range:' ,T35,2F12.3,' MeV')
        WRITE(NUNIT,1520) RLC(MEDIUM),RHO(MEDIUM),200.*TEFF0(1)
1520    FORMAT(/' Radiation length = ',T35,F13.6,' cm'/ ' DENSITY = ' ,T
     *35,F13.6,' g/cm**3'/ ' 200.*TEFFO = ' ,T35,F13.6,' cm')
        IF ((IAPRIM(1) .EQ. 0)) THEN
          WRITE(NUNIT,1530)
1530      FORMAT(/' Koch and Motz radiative stopping' )
        ELSE IF((IAPRIM(1) .EQ. 1)) THEN
          WRITE(NUNIT,1540)
1540      FORMAT(/' ICRU radiative stopping')
        ELSE
          WRITE(NUNIT,1550)
1550      FORMAT(/' No corrections in radiative cross sections ')
        END IF
        IF ((EPSTFL(1) .EQ. 0)) THEN
          WRITE(NUNIT,1560)
1560      FORMAT(' Standard PEGS4 density effect for collision stopping'
     *)
        ELSE
          WRITE(NUNIT,1570)
        END IF
1570    FORMAT(' ICRU or other density effect  (EPSTFL not zero)' )
        IF ((IUNRST(1) .EQ. 0)) THEN
          WRITE(NUNIT,1580)
1580      FORMAT(' Standard PEGS4 data set (restricted collision +', ' r
     *adiative (IUNRST=0))' )
        ELSE IF((IUNRST(1) .EQ. 1)) THEN
          WRITE(NUNIT,1590)
1590      FORMAT(' Unrestricted collision stopping power (IUNRST=1)')
        ELSE IF((IUNRST(1) .EQ. 2)) THEN
          WRITE(NUNIT,1600)
1600      FORMAT(' Unrestricted collison + radiative stopping power', '(
     *IUNRST=2)' )
        ELSE IF((IUNRST(1) .EQ. 3)) THEN
          WRITE(NUNIT,1610)
1610      FORMAT(' Unrestricted collision + restricted radiative', ' sto
     *pping power (IUNRST=3)' )
        ELSE IF((IUNRST(1) .EQ. 4)) THEN
          WRITE(NUNIT,1620)
1620      FORMAT(' Restricted collision + unrestricted radiative', ' sto
     *pping power (IUNRST=4)' )
        ELSE IF((IUNRST(1) .EQ. 5)) THEN
          WRITE(NUNIT,1630)
1630      FORMAT(' Unrestricted radiative stopping power (IUNRST=5)' )
        END IF
        IF ((ESTEPE .GT. 0.0)) THEN
          WRITE(NUNIT,1640) ESTEPE
        END IF
        WRITE(NUNIT,1650)
1650    FORMAT(///' Kin En',T13,'e(-),     e(+) dE/drhoX' ,T38,'e(-),e(+
     *) Mean free path (brem fraction)', /T5,'MeV',T17,'MeV/(g/cm**2)',T
     *56,'cm')
        IFLAG1=0
        IFLAG2=0
        DO 1661 I1=1,8
          DO 1671 I2=1,16
1680        IF ((INDIV .EQ. 1)) THEN
              WRITE(6,1690)
1690          FORMAT(' Electron kinetic energy( MeV): ',$)
              READ(5,'(F12.0)',ERR=1680,END= 1430) EKE
              IF ((EKE .GT. UE(1)-PRM .OR. EKE .LT. AE(1)-PRM)) THEN
                WRITE(6,1700)AE(1)-PRM,UE(1)-PRM
1700            FORMAT(T20,'   must be in range',2(1Pe10.3))
              END IF
              IF ((EKE .EQ. 0.0)) THEN
                STOP
              END IF
            ELSE
              EKE=ETAB(I2)*10.**(I1-4)
            END IF
            IF ((EKE .LE. AE(1)-PRM)) THEN
              IF ((IFLAG1 .EQ. 0)) THEN
                IFLAG1=1
                EKE=AE(1)-PRM
              ELSE
                EKE=0.0
              END IF
            END IF
            IF ((EKE .GT. UE(1)-PRM)) THEN
              IF ((IFLAG2 .EQ. 0)) THEN
                IFLAG2=1
                EKE=UE(1)-PRM
              ELSE
                EKE=1.E30
              END IF
            END IF
            EIE=EKE+PRM
            TMXS=0.0
            DEDXE=0.0
            DEDXP=0.0
            EFRACT=0.0
            IF ((EIE .GE. AE(1)-0.0001 .AND. EIE .LE. UE(1)+0.001)) THEN
              ELKE=LOG(EKE)
              LELKE=EKE1(MEDIUM)*ELKE+EKE0(MEDIUM)
              TMXS=TMXS1(LELKE,MEDIUM)*ELKE+TMXS0(LELKE,MEDIUM)
              DEDXE=EDEDX1(LELKE,MEDIUM)*ELKE+EDEDX0(LELKE,MEDIUM)
              DEDXP=PDEDX1(LELKE,MEDIUM)*ELKE+PDEDX0(LELKE,MEDIUM)
              EFRACT= TMXS*DEDXE/EKE
              SIGE=ESIG1(LELKE,MEDIUM)*ELKE+ESIG0(LELKE,MEDIUM)
              IF ((SIGE .EQ. 0.0)) THEN
                SIGE=1.E-10
              END IF
              SIGP=PSIG1(LELKE,MEDIUM)*ELKE+PSIG0(LELKE,MEDIUM)
              IF ((SIGP .EQ. 0.0)) THEN
                SIGP=1.E-10
              END IF
              BREME=EBR11(LELKE,MEDIUM)*ELKE+EBR10(LELKE,MEDIUM)
              BREMP=PBR11(LELKE,MEDIUM)*ELKE+PBR10(LELKE,MEDIUM)
              IF ((I2 .EQ. 1)) THEN
                WRITE(NUNIT,1710)
              END IF
              WRITE(NUNIT,1720) EKE,DEDXE/RHO(MEDIUM), DEDXP/RHO(MEDIUM)
     *        , 1./SIGE,BREME,1./SIGP,BREMP
1720          FORMAT(F9.3,T11,0PF9.4,T24,0PF9.4,T37, 1PE9.3,1x,'(',0PF7.
     *5,')',T59,1PE9.3,1x,'(',0PF7.5,')')
              IF ((IFILE .GE. 2)) THEN
                IPLOTE=IPLOTE+1
                PLOTEEN(IPLOTE)=EKE
                PLOTE(IPLOTE)=DEDXE/RHO(MEDIUM)
                PLOTEM(IPLOTE)=1./SIGE
                PLOTEMP(IPLOTE)=1./(SIGE*BREME)
                IF ((BREME .LT. 1.)) THEN
                  PLOTEMS(IPLOTE)=1./(SIGE*(1.-BREME))
                ELSE
                  PLOTEMS(IPLOTE)= 1e6
                END IF
              END IF
            END IF
1671      CONTINUE
1672      CONTINUE
1661    CONTINUE
1662    CONTINUE
      END IF
      IF ((IFILE .EQ. 1)) THEN
        CLOSE(NUNIT)
      END IF
      IF ((IFILE .GE. 2)) THEN
        TITLE=FILENAME(:lnblnk1(FILENAME))
1730    IF ((IOPT.EQ.2)) THEN
1741      CONTINUE
            WRITE(6,1750)
1750        FORMAT(/' Plot for photons (0) or exit (2):',$)
            READ(5,1760)PLTOPT
1760        FORMAT(I4)
            IF(((PLTOPT.EQ.0 .OR. PLTOPT.EQ.2)))GO TO1742
          GO TO 1741
1742      CONTINUE
        ELSE IF((IOPT.EQ.1)) THEN
1771      CONTINUE
            WRITE(6,1780)
1780        FORMAT(/' Plot for electrons (1) or exit (2):',$)
            READ(5,1790)PLTOPT
1790        FORMAT(I4)
            IF(((PLTOPT.EQ.1 .OR. PLTOPT.EQ.2)))GO TO1772
          GO TO 1771
1772      CONTINUE
        ELSE
1801      CONTINUE
            WRITE(6,1810)
1810        FORMAT(/' Plot for photons (0), electrons (1) or exit(2):',$
     *)
            READ(5,1820)PLTOPT
1820        FORMAT(I4)
            IF(((PLTOPT.EQ.0 .OR. PLTOPT.EQ.1 .OR. PLTOPT.EQ.2)))GO TO18
     *      02
          GO TO 1801
1802      CONTINUE
        END IF
        IF ((PLTOPT.EQ.0)) THEN
          IF ((IPLOTG.GT.0)) THEN
1831        CONTINUE
              WRITE(6,1840)
1840          FORMAT(/' Plot the relative components(0) or mfp(1):',$)
              READ(5,1850)IDO
1850          FORMAT(I10)
              IF(((IDO.EQ.0 .OR. IDO.EQ.1)))GO TO1832
            GO TO 1831
1832        CONTINUE
            IF ((IDO .EQ. 0)) THEN
              OUTFILE ='ph_comp_'//material
              FILELENGTH = 0
1861          CONTINUE
                FILELENGTH = FILELENGTH + 1
                IF(((OUTFILE(FILELENGTH:FILELENGTH) .EQ. ' ')))GO TO1862
              GO TO 1861
1862          CONTINUE
              OUTFILE(FILELENGTH:FILELENGTH+5)='.xvgr'
              SUBTITLE='Photons: Relative Components'
              OPEN(UNIT=7,FILE=OUTFILE,STATUS='UNKNOWN')
              SERIES='photoelectric component'
              ICURVE = 1
              CALL XVGRPLOT(PLOTGEN,PHOTE_COMP,DY,IPLOTG,(ICURVE-1),SERI
     *        ES, XAXIS,YAXISPcom,TITLE,SUBTITLE,7,0,HISTXMIN,2)
              SERIES='Compton component'
              ICURVE = 2
              CALL XVGRPLOT(PLOTGEN,COMP_COMP,DY,IPLOTG,(ICURVE-1),SERIE
     *        S, XAXIS,YAXISPcom,TITLE,SUBTITLE,7,0,HISTXMIN,2)
              SERIES='pair component'
              ICURVE = 3
              CALL XVGRPLOT(PLOTGEN,PAIR_COMP,DY,IPLOTG,(ICURVE-1),SERIE
     *        S, XAXIS,YAXISPcom,TITLE,SUBTITLE,7,0,HISTXMIN,2)
              IF ((IRAYLR(1) .EQ. 1)) THEN
                SERIES='Rayleigh/coherent component'
                ICURVE = 4
                CALL XVGRPLOT(PLOTGEN,COHER_COMP,DY,IPLOTG,(ICURVE-1),SE
     *          RIES, XAXIS,YAXISPcom,TITLE,SUBTITLE,7,0,HISTXMIN,2)
              END IF
              CLOSE(7)
              SCRIPT(3)='xmgrace '//OUTFILE//' 2> /dev/null'
              OPEN(4,FILE='xmgr_script',FORM='FORMATTED',STATUS='UNKNOWN
     *')
              WRITE(6,1870)SCRIPT(3)
1870          FORMAT(A80)
              WRITE (4,*)SCRIPT(1)(1:60)
              WRITE (4,*)SCRIPT(2)(1:60)
              WRITE (4,*)SCRIPT(3)
              CLOSE(4)
              istat = egs_system('chmod +x xmgr_script')
              istat = egs_system('./xmgr_script &')
            ELSE
              OUTFILE ='ph_mfp_'//material
              NCURVE = 1
              FILELENGTH = 0
1881          CONTINUE
                FILELENGTH = FILELENGTH + 1
                IF(((OUTFILE(FILELENGTH:FILELENGTH) .EQ. ' ')))GO TO1882
              GO TO 1881
1882          CONTINUE
              OUTFILE(FILELENGTH:FILELENGTH+5)='.xvgr'
              OPEN(UNIT=7,FILE=OUTFILE,STATUS='UNKNOWN')
              SERIES = ' photon mean free path'
              SUBTITLE='Photons:  Mean Free Path'
              CALL XVGRPLOT(PLOTGEN,PLOTG,DY,IPLOTG,0,SERIES, XAXIS,YAXI
     *        SPmfp,TITLE,SUBTITLE,7,0,HISTXMIN,0)
              CLOSE(7)
              SCRIPT(3)='xmgrace '//OUTFILE//' 2> /dev/null'
              OPEN(4,FILE='xmgr_script',FORM='FORMATTED',STATUS='UNKNOWN
     *')
              WRITE(6,1890)SCRIPT(3)
1890          FORMAT(A80)
              WRITE (4,*)SCRIPT(1)(1:60)
              WRITE (4,*)SCRIPT(2)(1:60)
              WRITE (4,*)SCRIPT(3)
              CLOSE(4)
              istat = egs_system('chmod +x xmgr_script')
              istat = egs_system('./xmgr_script &')
            END IF
          END IF
        ELSE IF((PLTOPT.EQ.1)) THEN
          NCURVE = 1
1901      CONTINUE
            WRITE(6,1910)
1910        FORMAT(/' Plot st.power(0), mfp to discrete int(1), brem(2),
     * sec.el(3):',$)
            READ(5,1920)EPOPT
1920        FORMAT(I4)
            IF(((EPOPT.GE.0 .AND. EPOPT.LE.3)))GO TO1902
          GO TO 1901
1902      CONTINUE
          IF ((EPOPT .EQ. 0)) THEN
            WRITE(6,1930)
1930        FORMAT('   doing stopping powers')
            IF ((IPLOTE.GT.0)) THEN
              OUTFILE ='e_sp_'//material
              SERIES='restricted stopping power'
              SUBTITLE='Electrons:  Stopping Powers'
              FILELENGTH = 0
1941          CONTINUE
                FILELENGTH = FILELENGTH + 1
                IF(((OUTFILE(FILELENGTH:FILELENGTH) .EQ. ' ')))GO TO1942
              GO TO 1941
1942          CONTINUE
              OUTFILE(FILELENGTH:FILELENGTH+5)='.xvgr'
              OPEN(UNIT=7,FILE=OUTFILE,STATUS='UNKNOWN')
              CALL XVGRPLOT(PLOTEEN,PLOTE,DY,IPLOTE,0,SERIES, XAXIS,YAXI
     *        SE,TITLE,SUBTITLE,7,0,HISTXMIN,2)
              CLOSE(7)
              SCRIPT(3)='xmgrace '//OUTFILE//' 2> /dev/null'
              OPEN(4,FILE='xmgr_script',FORM='FORMATTED',STATUS='UNKNOWN
     *')
              WRITE(6,1950)SCRIPT(3)
1950          FORMAT(A80)
              WRITE (4,*)SCRIPT(1)(1:60)
              WRITE (4,*)SCRIPT(2)(1:60)
              WRITE (4,*)SCRIPT(3)
              CLOSE(4)
              istat = egs_system('chmod +x xmgr_script')
              istat = egs_system('./xmgr_script &')
            END IF
          ELSE IF((EPOPT .EQ. 1)) THEN
            IF ((IPLOTE.GT.0)) THEN
              OUTFILE ='e_mfpd_'//material
              SERIES='mfp to discrete interaction'
              SUBTITLE='Electrons:  MFP to discrete interactions'
              FILELENGTH = 0
1961          CONTINUE
                FILELENGTH = FILELENGTH + 1
                IF(((OUTFILE(FILELENGTH:FILELENGTH) .EQ. ' ')))GO TO1962
              GO TO 1961
1962          CONTINUE
              OUTFILE(FILELENGTH:FILELENGTH+5)='.xvgr'
              OPEN(UNIT=7,FILE=OUTFILE,STATUS='UNKNOWN')
              CALL XVGRPLOT(PLOTEEN,PLOTEM,DY,IPLOTE,0,SERIES, XAXIS,YAX
     *        ISEmfp,TITLE,SUBTITLE,7,0,HISTXMIN,0)
              CLOSE(7)
              SCRIPT(3)='xmgrace '//OUTFILE//' 2> /dev/null'
              OPEN(4,FILE='xmgr_script',FORM='FORMATTED',STATUS='UNKNOWN
     *')
              WRITE(6,1970)SCRIPT(3)
1970          FORMAT(A80)
              WRITE (4,*)SCRIPT(1)(1:60)
              WRITE (4,*)SCRIPT(2)(1:60)
              WRITE (4,*)SCRIPT(3)
              CLOSE(4)
              istat = egs_system('chmod +x xmgr_script')
              istat = egs_system('./xmgr_script &')
            END IF
          ELSE IF((EPOPT .EQ. 2)) THEN
            IF ((IPLOTE.GT.0)) THEN
              OUTFILE ='e_mfpp_'//material
              SERIES='mfp to brem interaction'
              SUBTITLE='Electrons:  MFP to brem interactions'
              FILELENGTH = 0
1981          CONTINUE
                FILELENGTH = FILELENGTH + 1
                IF(((OUTFILE(FILELENGTH:FILELENGTH) .EQ. ' ')))GO TO1982
              GO TO 1981
1982          CONTINUE
              OUTFILE(FILELENGTH:FILELENGTH+5)='.xvgr'
              OPEN(UNIT=7,FILE=OUTFILE,STATUS='UNKNOWN')
              CALL XVGRPLOT(PLOTEEN,PLOTEMP,DY,IPLOTE,0,SERIES, XAXIS,YA
     *        XISPmfp,TITLE,SUBTITLE,7,0,HISTXMIN,0)
              CLOSE(7)
              SCRIPT(3)='xmgrace '//OUTFILE//' 2> /dev/null'
              OPEN(4,FILE='xmgr_script',FORM='FORMATTED',STATUS='UNKNOWN
     *')
              WRITE(6,1990)SCRIPT(3)
1990          FORMAT(A80)
              WRITE (4,*)SCRIPT(1)(1:60)
              WRITE (4,*)SCRIPT(2)(1:60)
              WRITE (4,*)SCRIPT(3)
              CLOSE(4)
              istat = egs_system('chmod +x xmgr_script')
              istat = egs_system('./xmgr_script &')
            END IF
          ELSE IF((EPOPT .EQ. 3)) THEN
            IF ((IPLOTE.GT.0)) THEN
              OUTFILE ='e_mfps_'//material
              SERIES='mfp to delta interaction'
              SUBTITLE='Electrons:  MFP to secondary electron interactio
     *ns'
              FILELENGTH = 0
2001          CONTINUE
                FILELENGTH = FILELENGTH + 1
                IF(((OUTFILE(FILELENGTH:FILELENGTH) .EQ. ' ')))GO TO2002
              GO TO 2001
2002          CONTINUE
              OUTFILE(FILELENGTH:FILELENGTH+5)='.xvgr'
              OPEN(UNIT=7,FILE=OUTFILE,STATUS='UNKNOWN')
              CALL XVGRPLOT(PLOTEEN,PLOTEMS,DY,IPLOTE,0,SERIES, XAXIS,YA
     *        XISPmfp,TITLE,SUBTITLE,7,0,HISTXMIN,2)
              CLOSE(7)
              SCRIPT(3)='xmgrace '//OUTFILE//' 2> /dev/null'
              OPEN(4,FILE='xmgr_script',FORM='FORMATTED',STATUS='UNKNOWN
     *')
              WRITE(6,2010)SCRIPT(3)
2010          FORMAT(A80)
              WRITE (4,*)SCRIPT(1)(1:60)
              WRITE (4,*)SCRIPT(2)(1:60)
              WRITE (4,*)SCRIPT(3)
              CLOSE(4)
              istat = egs_system('chmod +x xmgr_script')
              istat = egs_system('./xmgr_script &')
            END IF
          END IF
        END IF
        IF ((PLTOPT .NE. 2)) THEN
          GO TO 1730
        END IF
      END IF
      call egs_finish
      STOP
2020  FORMAT(///' WHAT FRACTION SHOULD TMXS BE(<=0.0 => exit): ')
1640  FORMAT(/' FIXTMX HAS ALTERED THE STEP ','SIZES USING ESTEPE=',F8.3
     *)
1710  FORMAT('   ')
      END
      SUBROUTINE AUSGAB(IARG)
      RETURN
      END
      SUBROUTINE HOWFAR
      RETURN
      END
      subroutine ranlux(rng_array)
      implicit none
      real*8 rng_array(24)
      integer*4 seedin,luxury_level
      integer*4 state(25)
      integer*4 ounit
      character*(*) fmt_flags
      integer*4 seeds(24),carry
      integer*4 i24,j24
      integer*4 next(24)
      integer*4 jseed_dflt,nskip,icon,j,k,status,jseed,nskipll(0:4),icar
     *ry
      logical not_initialized
      real*4 twom24,twop24
      integer*4 uni
      save seeds,carry,i24,j24,next,twom24,not_initialized, nskip,twop24
     *,nskipll
      data nskipll/0,24,73,199,365/
      data jseed_dflt/314159265/, icon/2147483563/
      data not_initialized/.true./
      IF (( not_initialized )) THEN
        not_initialized = .false.
        nskip = nskipll(1)
        twom24 = 1
        twop24 = 1
        jseed = jseed_dflt
        DO 2031 j=1,24
          twom24 = twom24 * 0.5
          twop24 = twop24 * 2
          k = jseed/53668
          jseed = 40014*(jseed-k*53668)-k*12211
          IF (( jseed .LT. 0 )) THEN
            jseed = jseed + icon
          END IF
          seeds(j) = mod(jseed,16777216)
          next(j) = j-1
2031    CONTINUE
2032    CONTINUE
        next(1) = 24
        i24 = 24
        j24 = 10
        carry = 0
        IF (( seeds(24) .EQ. 0 )) THEN
          carry = 1
        END IF
      END IF
      DO 2041 j=1,24
        uni = seeds(j24) - seeds(i24) - carry
        IF (( uni .LT. 0 )) THEN
          uni = uni + 16777216
          carry = 1
        ELSE
          carry = 0
        END IF
        seeds(i24) = uni
        i24 = next(i24)
        j24 = next(j24)
        IF (( uni .GE. 4096 )) THEN
          rng_array(j) = uni*twom24
        ELSE
          rng_array(j) = uni*twom24 + seeds(j24)*twom24*twom24
        END IF
2041  CONTINUE
2042  CONTINUE
      IF (( nskip .GT. 0 )) THEN
        DO 2051 j=1,nskip
          uni = seeds(j24) - seeds(i24) - carry
          IF (( uni .LT. 0 )) THEN
            uni = uni + 16777216
            carry = 1
          ELSE
            carry = 0
          END IF
          seeds(i24) = uni
          i24 = next(i24)
          j24 = next(j24)
2051    CONTINUE
2052    CONTINUE
      END IF
      return
      entry init_ranlux(luxury_level,seedin)
      jseed = seedin
      IF((jseed .LE. 0))jseed = jseed_dflt
      IF (( luxury_level .LT. 0 .OR. luxury_level .GT. 4 )) THEN
        luxury_level = 1
      END IF
      nskip = nskipll(luxury_level)
      WRITE(6,2060)luxury_level,jseed
2060  FORMAT(//' ***************** RANLUX initialization ***************
     ****'/, ' luxury level: ',i2,/, ' initial seed: ',i12,/, '*********
     ***************************************************'//)
      not_initialized = .false.
      twom24 = 1
      twop24 = 1
      DO 2071 j=1,24
        twom24 = twom24 * 0.5
        twop24 = twop24 * 2
        k = jseed/53668
        jseed = 40014*(jseed-k*53668)-k*12211
        IF (( jseed .LT. 0 )) THEN
          jseed = jseed + icon
        END IF
        seeds(j) = mod(jseed,16777216)
        next(j) = j-1
2071  CONTINUE
2072  CONTINUE
      next(1) = 24
      i24 = 24
      j24 = 10
      carry = 0.
      IF (( seeds(24) .EQ. 0 )) THEN
        carry = 1
      END IF
      return
      entry get_ranlux_state(state)
      DO 2081 j=1,24
        state(j) = seeds(j)
2081  CONTINUE
2082  CONTINUE
      state(25) = i24 + 100*(j24 + 100*nskip)
      IF((carry .GT. 0))state(25) = -state(25)
      return
      entry set_ranlux_state(state)
      twom24 = 1
      twop24 = 1
      DO 2091 j=1,24
        twom24 = twom24 * 0.5
        twop24 = twop24 * 2
        next(j) = j-1
2091  CONTINUE
2092  CONTINUE
      next(1) = 24
      DO 2101 j=1,24
        seeds(j) = state(j)
2101  CONTINUE
2102  CONTINUE
      IF (( state(25) .LE. 0 )) THEN
        status = -state(25)
        carry = 1
      ELSE
        status = state(25)
        carry = 0
      END IF
      nskip = status/10000
      status = status - nskip*10000
      j24 = status/100
      i24 = status - 100*j24
      IF (( j24 .LT. 1 .OR. j24 .GT. 24 .OR. i24 .LT. 1 .OR. i24 .GT. 24
     * )) THEN
        WRITE(6,2110)state(25),nskip,i24,j24
2110    FORMAT('// *********** Error in set_ranlux_state: seeds outsideo
     *f allowed range!'/, '   status = ',i8/, '   nskip  = ',i8/, '   i2
     *4    = ',i8/, '   j24    = ',i8/, '*******************************
     *****************************************'//)
        stop
      END IF
      not_initialized = .false.
      return
      entry show_ranlux_seeds(ounit)
      IF (( carry .GT. 0 )) THEN
        icarry = 1
      ELSE
        icarry = 0
      END IF
      write(ounit,'(a,i4,a,2i3,a,i2,$)') ' skip = ',nskip,' ix jx = ',i2
     *4,j24,' carry = ',icarry
      return
      entry print_ranlux_seeds(ounit,fmt_flags)
      IF (( carry .GT. 0 )) THEN
        icarry = 1
      ELSE
        icarry = 0
      END IF
      write(ounit,fmt_flags) nskip,i24,j24,icarry
      return
      end
      SUBROUTINE XVGRPLOT (X, Y, ERRY, NPTS, CURVENUM, SERIESTITLE, XTIT
     *LE, YTITLE, GRAPHTITLE, SUBTITLE, UNITNUM, TYPE, HISTXMIN, AXISTYP
     *E)
      IMPLICIT NONE
      integer*4 MAX, IDEBUG
      PARAMETER (MAX = 400)
      integer*4 NPTS,NPTS1,CURVENUM, COUNT,UNITNUM,TYPE,AXISTYPE
      real*8 X(NPTS),Y(NPTS),ERRY(NPTS),YMIN,HISTXMIN,ERRYOLD,SMALLESTX,
     * SMALLESTY,FUDGE
      integer*4 TITLELENGTH,SUBLENGTH,XAXISLENGTH,YAXISLENGTH,SERIESLENG
     *TH
      integer*4 LOGX, LOGY, LOGDY,ZEROYCOUNT
      CHARACTER*(*) SUBTITLE
      CHARACTER*(*) GRAPHTITLE,XTITLE,YTITLE,SERIESTITLE
      CHARACTER*10 INDEX
      CHARACTER*3 INDEXNUM
      logical TESTFILE, ALLPOS
      real*8 one
      parameter (one = 1)
      FUDGE = 1.e-10
      IDEBUG = 0
      IF ((IDEBUG .EQ. 1)) THEN
        write(6,'(//'' Entering xvgrplot ''/)')
        write(6,'('' Curve'',I3,'' to go to unit'',I3)')CURVENUM,UNITNUM
        write(6,'(''SERIESTITLE: '',a60)') SERIESTITLE
        write(6,'(''XTITLE:      '',a60)') XTITLE
        write(6,'(''YTITLE       '',a60)') YTITLE
        write(6,'(''GRAPHTILE:   '',a60)') graphtitle
        write(6,'(''SUBTITLE:    '',a80)') subtitle
      END IF
      IF (( NPTS .gt. MAX)) THEN
        WRITE(6,2120)NPTS, MAX
2120    FORMAT(//' **************************'/ ' Number of points asked
     * for =', I5, ' is greater than max allowed of', I4/ ' Setting NPTS
     * to MAX, you could adjust MAX in xvgrplot.mortran'/ ' ************
     ***************'//)
        NPTS1 = MAX
      ELSE
        NPTS1 = NPTS
      END IF
      INQUIRE(UNIT = UNITNUM,OPENED=TESTFILE)
      IF ((.NOT.TESTFILE)) THEN
        WRITE(6,2130) UNITNUM
2130    FORMAT (//'  ---------Error in Subroutine XVGRPLOT---------' ,/'
     *   Unit specified (',I2,') is not open.' ,/'   Unit must be opened
     * before using subroutine.' ,/'   Data not written to file.' ,/'  -
     *---------------------------------------------'//)
        RETURN
      END IF
      IF ((GRAPHTITLE .EQ. ' ')) THEN
        GRAPHTITLE = 'Untitled Graph - No title specified in subroutine'
      END IF
      IF ((XTITLE .EQ. ' ')) THEN
        XTITLE = 'X-axis not titled in subroutine'
      END IF
      IF ((YTITLE .EQ. ' ')) THEN
        YTITLE = 'Y-axis not titled in subroutine'
      END IF
      IF ((SERIESTITLE .EQ. ' ')) THEN
        SERIESTITLE = 'series # '
        INDEX = '0123456789'
        INDEXNUM = INDEX(CURVENUM+1:CURVENUM+1)
        SERIESTITLE(9:9) = INDEXNUM
      END IF
      TITLELENGTH = 61
      SUBLENGTH = 61
      XAXISLENGTH = 61
      YAXISLENGTH = 61
      SERIESLENGTH = 61
2141  CONTINUE
        TITLELENGTH = TITLELENGTH - 1
        IF(((GRAPHTITLE(TITLELENGTH:TITLELENGTH) .NE. ' ')))GO TO2142
      GO TO 2141
2142  CONTINUE
2151  CONTINUE
        SUBLENGTH = SUBLENGTH - 1
        IF(((SUBTITLE(SUBLENGTH:SUBLENGTH) .NE. ' ')))GO TO2152
      GO TO 2151
2152  CONTINUE
2161  CONTINUE
        XAXISLENGTH = XAXISLENGTH - 1
        IF(((XTITLE(XAXISLENGTH:XAXISLENGTH) .NE. ' ')))GO TO2162
      GO TO 2161
2162  CONTINUE
2171  CONTINUE
        YAXISLENGTH = YAXISLENGTH - 1
        IF(((YTITLE(YAXISLENGTH:YAXISLENGTH) .NE. ' ')))GO TO2172
      GO TO 2171
2172  CONTINUE
2181  CONTINUE
        SERIESLENGTH = SERIESLENGTH - 1
        IF(((SERIESTITLE(SERIESLENGTH:SERIESLENGTH) .NE. ' ')))GO TO2182
      GO TO 2181
2182  CONTINUE
      IF (( IDEBUG .EQ. 1)) THEN
        WRITE(6,2190)SERIESLENGTH,YAXISLENGTH,XAXISLENGTH,SUBLENGTH,TITL
     *  ELENGTH
2190    FORMAT(' SERIESLENGTH,YAXISLENGTH,XAXISLENGTH,SUBLENGTH,TITLELEN
     *GTH'/ 5I10)
      END IF
      LOGX = 0
      LOGY = 0
      LOGDY = 0
      ZEROYCOUNT=0
      ALLPOS=.TRUE.
      IF (( X(1).EQ.0.0 )) THEN
        SMALLESTX = 0.1
      ELSE
        SMALLESTX=X(1)
      END IF
      IF (( Y(1).EQ.0.0 )) THEN
        SMALLESTY = 0.1
      ELSE
        SMALLESTY=Y(1)
      END IF
      DO 2201 COUNT=1,NPTS1
        IF (((X(COUNT) .LT. SMALLESTX) .AND. (X(COUNT).NE.0.))) THEN
          SMALLESTX=X(COUNT)
        END IF
        IF (((Y(COUNT) .LT. SMALLESTY) .AND. (Y(COUNT).NE.0.))) THEN
          SMALLESTY=Y(COUNT)
        END IF
        IF (((X(COUNT) .LT. 0.).OR.(Y(COUNT) .LT. 0.))) THEN
          ALLPOS=.FALSE.
        END IF
2201  CONTINUE
2202  CONTINUE
      IF ((ALLPOS)) THEN
        DO 2211 COUNT=1,NPTS1
          IF ((X(COUNT).EQ.0.)) THEN
            X(COUNT)=SMALLESTX*FUDGE
          END IF
          IF ((Y(COUNT).EQ.0.)) THEN
            Y(COUNT)=SMALLESTY*FUDGE
          END IF
2211    CONTINUE
2212    CONTINUE
      END IF
      IF ((AXISTYPE .GT. 0)) THEN
        DO 2221 COUNT=1,NPTS1
          IF ((X(COUNT) .LE. 0.)) THEN
            LOGX = 1
          END IF
          IF ((Y(COUNT) .LE. 0.)) THEN
            LOGY = 1
          END IF
          IF (((Y(COUNT)-ERRY(COUNT)) .LE. 0.)) THEN
            LOGDY = 1
          END IF
2221    CONTINUE
2222    CONTINUE
      ELSE
        DO 2231 COUNT=1,NPTS1
          IF ((Y(COUNT)-ERRY(COUNT) .LT. Y(COUNT)-0.9999*Y(COUNT) .AND.
     *    Y(COUNT).GT.0.)) THEN
            ZEROYCOUNT=ZEROYCOUNT+1
            IF ((ZEROYCOUNT.EQ.1)) THEN
              WRITE(6,2240)
2240          FORMAT (/' ------------WARNING from Subroutine XVGRPLOT---
     *--------' ,/'  Some errors give 0 value and so are adjusted so   '
     * ,/'  that you can switch to a log Y scale while in xmgr.   ',/' -
     *------------------------------------------------------'/)
            END IF
            ERRYOLD=ERRY(COUNT)
            ERRY(COUNT) = 0.9999*Y(COUNT)
          END IF
2231    CONTINUE
2232    CONTINUE
      END IF
      IF ((CURVENUM .EQ. 0)) THEN
        IF ((AXISTYPE .EQ. 0)) THEN
          WRITE(UNITNUM,2250) 'xy'
        ELSE IF((AXISTYPE .EQ. 1)) THEN
          WRITE(UNITNUM,2250) 'logy'
          WRITE(UNITNUM,2260)
        ELSE IF((AXISTYPE .EQ. 2)) THEN
          WRITE(UNITNUM,2250) 'logx'
          WRITE(UNITNUM,2260)
        ELSE IF((AXISTYPE .EQ. 3)) THEN
          WRITE(UNITNUM,2250) 'logxy'
          WRITE(UNITNUM,2260)
          WRITE(UNITNUM,2270)
        ELSE
          WRITE(6,2280) AXISTYPE
2280      FORMAT (//'  ------------Error in Subroutine XVGRPLOT---------
     *--' ,/'   AXISTYPE specified (',I2,') is not a valid option.' ,/' 
     *----------------------------------------------'//)
          RETURN
        END IF
2250    FORMAT ('@g0 type ',A,' ')
2260    FORMAT ('@    xaxis  ticklabel format exponential')
2270    FORMAT ('@    yaxis  ticklabel format exponential')
        WRITE(UNITNUM,2290) GRAPHTITLE(1:TITLELENGTH) ,SUBTITLE(1:SUBLEN
     *  GTH) ,XTITLE(1:XAXISLENGTH) ,YTITLE(1:YAXISLENGTH)
2290    FORMAT ('@    title "',A,'"'/ ,'@    subtitle "',A,'"'/ ,'@    l
     *egend on'/ ,'@    legend box linestyle 0'/ ,'@    legend x1 0.6'/,
     *'@    legend y1 0.75'/ ,'@    view xmin 0.250000'/ ,'@    xaxis  l
     *abel "',A,'"'/ ,'@    timestamp on'/ ,'@    yaxis  label "',A,'"')
      END IF
      IF ((AXISTYPE .EQ. 1 .AND. LOGY .EQ. 1)) THEN
        WRITE(UNITNUM,2250) 'xy'
        WRITE(6,2300)
2300    FORMAT (/' ----------WARNING from Subroutine XVGRPLOT---------',
     */'  Log scale requested for Y axis when one or more   ' ,/'  Ydata
     * points are 0 or negative.                  ' ,//'  Y axis scale c
     *hanged to linear.                   ' ,/' ------------------------
     *---------------------------'/)
      END IF
      IF ((AXISTYPE .EQ. 2 .AND. LOGX .EQ. 1)) THEN
        WRITE(UNITNUM,2250) 'xy'
        WRITE(6,2310)
2310    FORMAT (/' ----------WARNING from Subroutine XVGRPLOT---------',
     */'  Log scale requested for X axis when one or more   ' ,/'  Xdata
     * points are 0 or negative.                  ' ,//'  X axis scale c
     *hanged to linear.                   ' ,/' ------------------------
     *---------------------------'/)
      END IF
      IF ((AXISTYPE .EQ. 3 .AND. (LOGX .EQ. 1 .OR. LOGY .EQ. 1))) THEN
        IF ((LOGX .EQ. 1 .AND. LOGY .EQ. 1)) THEN
          WRITE(UNITNUM,2250) 'xy'
          WRITE(6,2320)
2320      FORMAT (/' ----------WARNING from Subroutine XVGRPLOT---------
     *' ,/'  Log scale requested for X axis and Y axis when    ' ,/'  on
     *e or more X and Y data points are 0 or negative.' ,//'  X and Y ax
     *es scales changed to linear.            ' ,/' --------------------
     *-------------------------------'/)
        ELSE IF((LOGX .EQ. 1)) THEN
          WRITE(UNITNUM,2250) 'logy'
          WRITE(6,2310)
        ELSE
          WRITE(UNITNUM,2250) 'logx'
          WRITE(6,2300)
        END IF
      END IF
      IF ((LOGDY .EQ. 1 .AND. LOGY .NE. 1 .AND. (AXISTYPE .EQ. 3 .OR. AX
     *ISTYPE .EQ. 1))) THEN
        WRITE(6,2330)
2330    FORMAT (/' ------------WARNING from Subroutine XVGRPLOT---------
     *--' ,/'  Log scale requested for Y axis, and Y value less      ' ,
     */'  error gives 0 or negative value.                      ' ,//'  
     *Error adjusted to aviod negavite values on log scale. ' ,/' ------
     *-------------------------------------------------'/)
        DO 2341 COUNT=1,NPTS1
          IF ((Y(COUNT)-ERRY(COUNT) .LT. Y(COUNT)-0.9999*Y(COUNT))) THEN
            ERRYOLD = ERRY(COUNT)
            ERRY(COUNT) = 0.9999*Y(COUNT)
            WRITE(6,2350) COUNT,ERRYOLD,ERRY(COUNT)
2350        FORMAT (/'  Error adjusted on point #',I2,' from',1PE11.4, '
     * to' /'        ',1PE11.4,'.')
          END IF
2341    CONTINUE
2342    CONTINUE
        WRITE(6,2360)
2360    FORMAT (/' -----------------------------------------------------
     *--'/)
      END IF
      IF ((CURVENUM .LT. 10 )) THEN
        WRITE(UNITNUM,'(''@    s'',I1,'' on'')') CURVENUM
      ELSE
        WRITE(UNITNUM,'(''@    s'',I2,'' on'')') CURVENUM
      END IF
      WRITE(UNITNUM,2370) CURVENUM,SERIESTITLE(1:SERIESLENGTH)
2370  FORMAT ('@    legend string ',I2,' "',A,'"')
      IF ((TYPE .EQ. 0)) THEN
        DO 2381 COUNT=1,NPTS1
          IF ((ERRY(COUNT) .NE. 0)) THEN
            GOTO 2390
          END IF
2381    CONTINUE
2382    CONTINUE
        WRITE(UNITNUM,2400)
2400    FORMAT ('@TYPE xy')
        IF ((CURVENUM .LT. 10)) THEN
          WRITE(UNITNUM,2410) CURVENUM
          IF ((CURVENUM .EQ. 9)) THEN
            WRITE(UNITNUM,2420) CURVENUM, CURVENUM+1
          ELSE
            WRITE(UNITNUM,2430) CURVENUM, CURVENUM+1
          END IF
        ELSE
          WRITE(UNITNUM,2440) CURVENUM
          WRITE(UNITNUM,2450) CURVENUM, CURVENUM+1
        END IF
2410    FORMAT ('@    s',I1,' errorbar length 0.000000')
2440    FORMAT ('@    s',I2,' errorbar length 0.000000')
2420    FORMAT ('@    s',I1,' symbol color ',I2)
2430    FORMAT ('@    s',I1,' symbol color ',I1)
2450    FORMAT ('@    s',I2,' symbol color ',I2)
        DO 2461 COUNT=1,NPTS1
          WRITE(UNITNUM,2470) X(COUNT),Y(COUNT)
2461    CONTINUE
2462    CONTINUE
2470    FORMAT (1PE15.4,1PE15.4)
        GOTO 2480
2390    CONTINUE
        WRITE(UNITNUM,2490)
2490    FORMAT ('@TYPE xydy')
        IF ((CURVENUM .LT. 10)) THEN
          WRITE(UNITNUM,2410) CURVENUM
          IF ((CURVENUM .EQ. 9)) THEN
            WRITE(UNITNUM,2420) CURVENUM, CURVENUM+1
          ELSE
            WRITE(UNITNUM,2430) CURVENUM, CURVENUM+1
          END IF
        ELSE
          WRITE(UNITNUM,2440) CURVENUM
          WRITE(UNITNUM,2450) CURVENUM, CURVENUM+1
        END IF
        DO 2501 COUNT=1,NPTS1
          WRITE(UNITNUM,2510) X(COUNT),Y(COUNT),ERRY(COUNT)
2501    CONTINUE
2502    CONTINUE
2510    FORMAT (1PE15.4,1PE15.4,1PE15.4)
2480    CONTINUE
      ELSE
        YMIN = ABS(1.E5 * Y(1))
        DO 2521 COUNT=1,NPTS1
          IF ((ABS(Y(COUNT)) .LT. YMIN)) THEN
            YMIN = ABS(Y(COUNT))
          END IF
2521    CONTINUE
2522    CONTINUE
        YMIN = SIGN(one,Y(1)) * 1.E-5 * YMIN
        Y(NPTS1+1) = YMIN
        IF (((AXISTYPE .EQ. 2 .OR. AXISTYPE .EQ. 3) .AND. HISTXMIN .EQ.
     *  0)) THEN
          IF ((X(1) .EQ. (X(2)-X(1)))) THEN
            HISTXMIN = X(1)-0.5*(X(2)-X(1))
          ELSE
            HISTXMIN = X(1)-(X(2)-X(1))
          END IF
          WRITE(6,2530) HISTXMIN
2530      FORMAT (/' ---------WARNING from Subroutine XVGRPLOT--------',
     */'  Minimum bin for X specified as 0 with log scale  ' ,/'  on X a
     *xis.  Minimum X bin set to ',1PE10.3,'.' ,/' ---------------------
     *----------------------------'/)
        END IF
        DO 2541 COUNT=1,NPTS1
          IF ((ERRY(COUNT) .NE. 0.)) THEN
            GOTO 2550
          END IF
2541    CONTINUE
2542    CONTINUE
        IF ((CURVENUM .LT. 10)) THEN
          WRITE(UNITNUM,2410) CURVENUM
          IF ((CURVENUM .EQ. 9)) THEN
            WRITE(UNITNUM,2420) CURVENUM, CURVENUM+1
          ELSE
            WRITE(UNITNUM,2430) CURVENUM, CURVENUM+1
          END IF
        ELSE
          WRITE(UNITNUM,2440) CURVENUM
          WRITE(UNITNUM,2450) CURVENUM, CURVENUM+1
        END IF
        WRITE(UNITNUM,2400)
        WRITE(UNITNUM,2470) HISTXMIN,YMIN
        WRITE(UNITNUM,2470) HISTXMIN,Y(1)
        DO 2561 COUNT=1,NPTS1
          WRITE(UNITNUM,2470) X(COUNT),Y(COUNT)
          WRITE(UNITNUM,2470) X(COUNT),Y(COUNT+1)
2561    CONTINUE
2562    CONTINUE
        GOTO 2570
2550    CONTINUE
        ERRY(NPTS1+1) = 0.0
        IF ((CURVENUM .LT. 10)) THEN
          WRITE(UNITNUM,2410) CURVENUM
          IF ((CURVENUM .EQ. 9)) THEN
            WRITE(UNITNUM,2420) CURVENUM, CURVENUM+1
          ELSE
            WRITE(UNITNUM,2430) CURVENUM, CURVENUM+1
          END IF
        ELSE
          WRITE(UNITNUM,2440) CURVENUM
          WRITE(UNITNUM,2450) CURVENUM, CURVENUM+1
        END IF
        WRITE(UNITNUM,2490)
        IF ((HISTXMIN .EQ. 0.0)) THEN
          HISTXMIN = SMALLESTX*FUDGE
        END IF
        WRITE(UNITNUM,2510) HISTXMIN, YMIN, 0.
        WRITE(UNITNUM,2510) HISTXMIN, Y(1), 0.
        WRITE(UNITNUM,2510) (X(1)+HISTXMIN)/2., Y(1), ERRY(1)
        DO 2581 COUNT=1,NPTS1
          WRITE (UNITNUM,2510) X(COUNT),Y(COUNT),0.
          WRITE (UNITNUM,2510) X(COUNT),Y(COUNT+1),0.
          IF ((COUNT .LT. NPTS1)) THEN
            WRITE (UNITNUM,2510) (X(COUNT)+X(COUNT+1))/2.,Y(COUNT+1),ERR
     *      Y(COUNT+1)
          END IF
2581    CONTINUE
2582    CONTINUE
2570    CONTINUE
      END IF
      WRITE(UNITNUM,'(''&'')')
      RETURN
      END
C##############################################################################
C
C   This file was automatically generated by configure version 2.0
C   It contains various subroutines and functions for date, time,
C   CPU time, host name, etc.
C
C   Attention: all changes will be lost the next time you run configure!
C
C##############################################################################


C##############################################################################
C
C  EGSnrc egs_system subroutine v1
C  Copyright (C) 2015 National Research Council Canada
C
C  This file is part of EGSnrc.
C
C  EGSnrc is free software: you can redistribute it and/or modify it under
C  the terms of the GNU Affero General Public License as published by the
C  Free Software Foundation, either version 3 of the License, or (at your
C  option) any later version.
C
C  EGSnrc is distributed in the hope that it will be useful, but WITHOUT ANY
C  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
C  FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for
C  more details.
C
C  You should have received a copy of the GNU Affero General Public License
C  along with EGSnrc. If not, see <http://www.gnu.org/licenses/>.
C
C##############################################################################
C
C  Author:          Iwan Kawrakow, 2003
C
C  Contributors:
C
C##############################################################################


C*****************************************************************************
C egs_system(command)  runs a system command and returns the status
C                      command must be null-terminated
C*****************************************************************************
      integer function egs_system(command)
      character*(*) command
      integer system, istat
      istat = system(command)
      egs_system = istat
      return
      end

C##############################################################################
C
C  EGSnrc egs_isdir subroutine v1
C  Copyright (C) 2015 National Research Council Canada
C
C  This file is part of EGSnrc.
C
C  EGSnrc is free software: you can redistribute it and/or modify it under
C  the terms of the GNU Affero General Public License as published by the
C  Free Software Foundation, either version 3 of the License, or (at your
C  option) any later version.
C
C  EGSnrc is distributed in the hope that it will be useful, but WITHOUT ANY
C  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
C  FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for
C  more details.
C
C  You should have received a copy of the GNU Affero General Public License
C  along with EGSnrc. If not, see <http://www.gnu.org/licenses/>.
C
C##############################################################################
C
C  Author:          Iwan Kawrakow, 2003
C
C  Contributors:
C
C##############################################################################


C*****************************************************************************
C
C  egs_isdir(file_name)  Returns .true., if the string file_name points to
C                        an existing directory. This version uses the lstat
C                        intrinsic and then tests for bit 14 being set in
C                        the mode element. This works on all Unix systems
C                        that I have access to (Linux, Aix, HP-UX, OSF1,
C                        Solaris, IRIX)
C
C*****************************************************************************

      logical function egs_isdir(file_name)
      implicit none
      character*(*) file_name
      integer*4 lnblnk1, res, array(13), l, lstat
      logical btest
      egs_isdir = .false.
      l = lnblnk1(file_name)
      if( l.lt.len(file_name) ) file_name(l+1:l+1) = char(0)
         ! On some systems lstat only works if the string is 0-terminated
      res = lstat(file_name,array)
      if( l.lt.len(file_name) ) file_name(l+1:l+1) = ' '
      if( res.eq.0 ) then
            ! Amost all compilers that have the lstat intrinsic return the
            ! file mode in the 3rd array element. But the PGI compiler has
            ! its own opinion on the subject and returns it in the 5th element
            ! That's why the relevant element is written as 3
            ! here, 3 gets replaced by the appropriate element
            ! by the configure script.
          if( btest(array(3),14) ) egs_isdir = .true.
      end if
      return
      end

C##############################################################################
C
C  EGSnrc date subroutines v1
C  Copyright (C) 2015 National Research Council Canada
C
C  This file is part of EGSnrc.
C
C  EGSnrc is free software: you can redistribute it and/or modify it under
C  the terms of the GNU Affero General Public License as published by the
C  Free Software Foundation, either version 3 of the License, or (at your
C  option) any later version.
C
C  EGSnrc is distributed in the hope that it will be useful, but WITHOUT ANY
C  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
C  FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for
C  more details.
C
C  You should have received a copy of the GNU Affero General Public License
C  along with EGSnrc. If not, see <http://www.gnu.org/licenses/>.
C
C##############################################################################
C
C  Author:          Iwan Kawrakow, 2003
C
C  Contributors:
C
C##############################################################################


C***************************************************************************
C
C   egs_fdate(out):  print a 24 char date and time string in the form
C                         'Tue Mar 18 08:16:42 2003'
C                    to the unit specified by out without end of line
C                    i.e. the sequence
C                    write(6,'(a,$)') 'Today is '
C                    call egs_fdate(6)
C                    write(6,'(a)') '. Have a nice date'
C                    should result in something like
C                    Today is Tue Mar 18 08:16:42 2003. Have a nice date
C                    printed to unit 6.
C
C***************************************************************************

      subroutine egs_fdate(ounit)
      integer ounit
      character*24 string
      call fdate(string)
      write(ounit,'(a,$)') string
      end

C***************************************************************************
C
C   egs_get_fdate(string) assignes a 24 char date and time string to string
C                         string must be at least 24 chars long, otherwise
C                         this subroutine has no effect.
C
C***************************************************************************

      subroutine egs_get_fdate(string)
      character*(*) string
      if( len(string).ge.24 ) call fdate(string)
      return
      end

C##############################################################################
C
C  EGSnrc egs_date_and_time subroutine v1
C  Copyright (C) 2015 National Research Council Canada
C
C  This file is part of EGSnrc.
C
C  EGSnrc is free software: you can redistribute it and/or modify it under
C  the terms of the GNU Affero General Public License as published by the
C  Free Software Foundation, either version 3 of the License, or (at your
C  option) any later version.
C
C  EGSnrc is distributed in the hope that it will be useful, but WITHOUT ANY
C  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
C  FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for
C  more details.
C
C  You should have received a copy of the GNU Affero General Public License
C  along with EGSnrc. If not, see <http://www.gnu.org/licenses/>.
C
C##############################################################################
C
C  Author:          Iwan Kawrakow, 2003
C
C  Contributors:
C
C##############################################################################


      subroutine egs_date_and_time(vnow)
      integer vnow(8)
      character dat*8,tim*10,zon*5
      call date_and_time(dat,tim,zon,vnow)
      return
      end

C##############################################################################
C
C  EGSnrc egs_date subroutine v1
C  Copyright (C) 2015 National Research Council Canada
C
C  This file is part of EGSnrc.
C
C  EGSnrc is free software: you can redistribute it and/or modify it under
C  the terms of the GNU Affero General Public License as published by the
C  Free Software Foundation, either version 3 of the License, or (at your
C  option) any later version.
C
C  EGSnrc is distributed in the hope that it will be useful, but WITHOUT ANY
C  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
C  FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for
C  more details.
C
C  You should have received a copy of the GNU Affero General Public License
C  along with EGSnrc. If not, see <http://www.gnu.org/licenses/>.
C
C##############################################################################
C
C  Author:          Iwan Kawrakow, 2003
C
C  Contributors:
C
C##############################################################################


C*************************************************************************
C
C egs_date(ounit): print a 11 char string in the form
C                     '18-Mar-2003'
C                  to the unit specified by ounit
C                  No end of line character is inserted
C
C*************************************************************************

      subroutine egs_date(ounit)
      integer ounit
      character string*24, dat*11
      call fdate(string)
      dat(1:2) = string(9:10)
      dat(3:3) = '-'
      dat(4:6) = string(5:7)
      dat(7:7) = '-'
      dat(8:11) = string(21:24)
      write(ounit,'(a,$)') dat
      return
      end

C##############################################################################
C
C  EGSnrc egs_time subroutine v1
C  Copyright (C) 2015 National Research Council Canada
C
C  This file is part of EGSnrc.
C
C  EGSnrc is free software: you can redistribute it and/or modify it under
C  the terms of the GNU Affero General Public License as published by the
C  Free Software Foundation, either version 3 of the License, or (at your
C  option) any later version.
C
C  EGSnrc is distributed in the hope that it will be useful, but WITHOUT ANY
C  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
C  FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for
C  more details.
C
C  You should have received a copy of the GNU Affero General Public License
C  along with EGSnrc. If not, see <http://www.gnu.org/licenses/>.
C
C##############################################################################
C
C  Author:          Iwan Kawrakow, 2003
C
C  Contributors:
C
C##############################################################################


C $Id: egs_time_v1.f,v 1.1 2003/07/11 19:17:08 iwan Exp $
C*************************************************************************
C
C egs_time(ounit): print a 8 char string in the form hh:mm:ss
C                  to the unit specified by ounit
C                  No end of line character is inserted
C
C*************************************************************************

      subroutine egs_time(ounit)
      integer ounit
      character string*24
      call fdate(string)
      write(ounit,'(a,$)') string(12:19)
      return
      end

C##############################################################################
C
C  EGSnrc seconds timing subroutines v1
C  Copyright (C) 2015 National Research Council Canada
C
C  This file is part of EGSnrc.
C
C  EGSnrc is free software: you can redistribute it and/or modify it under
C  the terms of the GNU Affero General Public License as published by the
C  Free Software Foundation, either version 3 of the License, or (at your
C  option) any later version.
C
C  EGSnrc is distributed in the hope that it will be useful, but WITHOUT ANY
C  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
C  FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for
C  more details.
C
C  You should have received a copy of the GNU Affero General Public License
C  along with EGSnrc. If not, see <http://www.gnu.org/licenses/>.
C
C##############################################################################
C
C  Author:          Iwan Kawrakow, 2003
C
C  Contributors:
C
C##############################################################################


C*****************************************************************************
C
C real function egs_secnds(t0): returns seconds passed since midnight minus t0
C
C*****************************************************************************

      real function egs_secnds(t0)
      real t0,t1
      character dat*8,tim*10,zon*5
      integer values(8)
      call date_and_time(dat,tim,zon,values)
      t1 = 3600.*values(5) + 60.*values(6) + values(7) + 0.001*values(8)
      egs_secnds = t1 - t0
      return
      end

C*****************************************************************************
C
C real function egs_tot_time()
C
C   On first call returns seconds passed since 1/1/1970
C   On subsequent calls returns
C     - seconds since last call, if flag = 0
C     - seconds since first call, else
C
C*****************************************************************************

      real function egs_tot_time(flag)
      integer flag
      character dat*8,tim*10,zon*5
      integer vnow(8), vlast(8),i
      real t,egs_time_diff,t0
      data vlast/1970,1,1,5*0/,t0/-1/
      save vlast,t0
      call date_and_time(dat,tim,zon,vnow)
      t = egs_time_diff(vlast,vnow)
      do i=1,8
        vlast(i)=vnow(i)
      end do
      if( t0.lt.0 ) then
        t0 = 0
        egs_tot_time = t
      else
        t0 = t0 + t
        if(flag.eq.0) then
          egs_tot_time = t
        else
          egs_tot_time = t0
        end if
      end if
      return
      end

C##############################################################################
C
C  EGSnrc date and time subroutines
C  Copyright (C) 2015 National Research Council Canada
C
C  This file is part of EGSnrc.
C
C  EGSnrc is free software: you can redistribute it and/or modify it under
C  the terms of the GNU Affero General Public License as published by the
C  Free Software Foundation, either version 3 of the License, or (at your
C  option) any later version.
C
C  EGSnrc is distributed in the hope that it will be useful, but WITHOUT ANY
C  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
C  FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for
C  more details.
C
C  You should have received a copy of the GNU Affero General Public License
C  along with EGSnrc. If not, see <http://www.gnu.org/licenses/>.
C
C##############################################################################
C
C  Author:          Iwan Kawrakow, 2003
C
C  Contributors:
C
C##############################################################################


C****************************************************************************
C
C Returns the time difference between vstart and vend
C vstart and vend are integer arrays of dimension 8 with elements
C corresponding to the specification of the data_and_time routine, i.e.
C   array(1) = year
C   array(2) = month of the year   (1...12)
C   array(3) = day of the month    (1...31)
C   array(4) = difference in minutes from UTC
C   array(5) = hour of the day     (1...23)
C   array(6) = minute of the hour  (1...59)
C   array(7) = seconds of the minute (1...59)
C   array(8) = miliseconds of the second (1...999)
C
C Note: this implementation ignores the time difference from UTC field
C
C*****************************************************************************
      real function egs_time_diff(vstart,vend)
      integer    vstart(8),vend(8)
      real       egs_time_diff_o
      if( vend(1).lt.vstart(1).or.
     &  (vend(1).eq.vstart(1).and.vend(2).lt.vstart(2)) ) then
        egs_time_diff = -egs_time_diff_o(vend,vstart)
      else
        egs_time_diff = egs_time_diff_o(vstart,vend)
      end if
      return
      end

C******************************************************************************
C
C day difference between the dates specified by the integer arrays vstart and
C vend. The arrays are v(1)=year, v(2)=month, v(3)=day
C
C******************************************************************************
      integer function egs_day_diff(vstart,vend)
      integer vstart(3),vend(3),egs_day_diff_o
      if( vend(1).lt.vstart(1).or.
     &  (vend(1).eq.vstart(1).and.vend(2).lt.vstart(2)) ) then
        egs_day_diff = -egs_day_diff_o(vend,vstart)
      else
        egs_day_diff = egs_day_diff_o(vstart,vend)
      end if
      return
      end

C******************************************************************************
C
C Returns a 3-letter abreviation of the day of the week in the string day,
C given a day specified by the integer array values
C   values(1)=year, values(2)=month, values(3)=day
C
C******************************************************************************
      subroutine egs_weekday(values,day)
      character*(*) day
      integer       values(3)
      integer       days,vtmp(3),egs_day_diff,aux
      character*3   wdays(7)
      data wdays/'Mon','Tue','Wed','Thu','Fri','Sat','Sun'/
      vtmp(1) = 1970
      vtmp(2) = 1
      vtmp(3) = 1
      days = egs_day_diff(vtmp,values)
      aux = mod(days,7)
      days = 4 + aux
      if( days.gt.7 ) days = days - 7
      day(:len(day)) = ' '
      aux = min(len(day),3)
      day(:aux) = wdays(days)(:aux)
      return
      end

C*****************************************************************************
C
C Same as egs_day_diff above, but assumes that vend specifies a later date
C than vstart.
C
C*****************************************************************************
      integer function egs_day_diff_o(vstart,vend)
      integer vstart(3),vend(3)
      integer    days
      logical    next_month
      integer    tm,m,ty,y
      integer    mdays(12)
      data       mdays/31,28,31,30,31,30,31,31,30,31,30,31/
      days = 0
      ty = vstart(1)
      y  = vend(1)
      tm = vstart(2)
      m  = vend(2)
      next_month = .true.
      do while(next_month)
        if( tm.eq.m.and.ty.eq.y ) then
          next_month = .false.
        else
          days = days + mdays(tm)
          if( tm.eq.2.and.mod(ty,4).eq.0 ) days = days + 1
          tm = tm + 1
          if( tm.gt.12 ) then
            ty = ty + 1
            tm = 1
          end if
        end if
      end do
      days = days + vend(3) - vstart(3)
      egs_day_diff_o = days
      return
      end

C******************************************************************************
C
C Same as egs_time_diff above, but assumes that vend specifies a later date
C than vstart.
C
C******************************************************************************
      real function egs_time_diff_o(vstart,vend)
      integer    vstart(8),vend(8)
      integer    days,hours,minutes,secs,msecs
      integer    egs_day_diff_o
      days = egs_day_diff_o(vstart,vend)
      hours = vend(5) - vstart(5)
      minutes = vend(6) - vstart(6)
      secs = vend(7) - vstart(7)
      msecs = vend(8) - vstart(8)
      egs_time_diff_o = 3600.*(24.*days+hours)+60.*minutes+secs+
     &                  0.001*msecs
      return
      end

C******************************************************************************
C
C Returns in month a 3-letter abreviation of the month specified by mo, if
C mo is between 1 and 12, or an empty string otherwise.
C
C******************************************************************************
      subroutine egs_month(mo,month)
      integer mo
      character*(*) month
      integer iaux
      character*3   months(12)
      data months/'Jan','Feb','Mar','Apr','May','Jun', 'Jul','Aug','Sep'
     *,'Oct','Nov','Dec'/
      iaux = min(len(month),3)
      month(:len(month)) = ' '
      if( mo.ge.1.and.mo.le.12 ) month(:iaux) = months(mo)(:iaux)
      return
      end

C******************************************************************************
C
C Converts a 3-letter abreviation of a month to its corresponding integer
C value, if the string month is a valid month, or -1 otherwise.
C
C******************************************************************************
      integer function egs_conver_month(month)
      character*3 month
      character*3 months(12)
      integer i
      data months/'Jan','Feb','Mar','Apr','May','Jun', 'Jul','Aug','Sep'
     *,'Oct','Nov','Dec'/
      do i=1,12
        if( month.eq.months(i) ) then
          egs_conver_month = i
          return
        end if
      end do
      egs_conver_month = -1
      return
      end


C##############################################################################
C
C  EGSnrc egs_etime subroutine
C  Copyright (C) 2015 National Research Council Canada
C
C  This file is part of EGSnrc.
C
C  EGSnrc is free software: you can redistribute it and/or modify it under
C  the terms of the GNU Affero General Public License as published by the
C  Free Software Foundation, either version 3 of the License, or (at your
C  option) any later version.
C
C  EGSnrc is distributed in the hope that it will be useful, but WITHOUT ANY
C  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
C  FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for
C  more details.
C
C  You should have received a copy of the GNU Affero General Public License
C  along with EGSnrc. If not, see <http://www.gnu.org/licenses/>.
C
C##############################################################################
C
C  Author:          Iwan Kawrakow, 2003
C
C  Contributors:
C
C##############################################################################


C*****************************************************************************
C
C real function egs_etime(): returns CPU time consumed since the start of
C                            the program
C
C*****************************************************************************

      real function egs_etime()
      real tarray(2),etime
      egs_etime = etime(tarray)
      return
      end

C##############################################################################
C
C  EGSnrc canonical system name subroutines
C  Copyright (C) 2015 National Research Council Canada
C
C  This file is part of EGSnrc.
C
C  EGSnrc is free software: you can redistribute it and/or modify it under
C  the terms of the GNU Affero General Public License as published by the
C  Free Software Foundation, either version 3 of the License, or (at your
C  option) any later version.
C
C  EGSnrc is distributed in the hope that it will be useful, but WITHOUT ANY
C  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
C  FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for
C  more details.
C
C  You should have received a copy of the GNU Affero General Public License
C  along with EGSnrc. If not, see <http://www.gnu.org/licenses/>.
C
C##############################################################################
C
C  Author:          Iwan Kawrakow, 2003
C
C  Contributors:
C
C##############################################################################


C******************************************************************************
C
C Print the canonical system name as determined by the config.guess script
C or the Windows installation program to the unit specified by ounit.
C
C*****************************************************************************

      subroutine egs_print_canonical_system(ounit)
      integer ounit
      write(6,'(a,$)') 'x86_64-unknown-linux-gnu'
      return
      end

C******************************************************************************
C
C Assign the canonical system name as determined by the config.guess script
C or the Windows installation program to the string pointed to by res
C
C******************************************************************************

      subroutine egs_get_canonical_system(res)
      character*(*) res
      integer l1,l2
      l1 = lnblnk1('x86_64-unknown-linux-gnu')
      l2 = len(res)
      res(:l2) = ' '
      if( l2.ge.l1 ) then
        res(:l1) = 'x86_64-unknown-linux-gnu'
      else
        res(:l2) = 'x86_64-unknown-linux-gnu'
      end if
      return
      end


C##############################################################################
C
C  EGSnrc configuration name subroutines
C  Copyright (C) 2015 National Research Council Canada
C
C  This file is part of EGSnrc.
C
C  EGSnrc is free software: you can redistribute it and/or modify it under
C  the terms of the GNU Affero General Public License as published by the
C  Free Software Foundation, either version 3 of the License, or (at your
C  option) any later version.
C
C  EGSnrc is distributed in the hope that it will be useful, but WITHOUT ANY
C  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
C  FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for
C  more details.
C
C  You should have received a copy of the GNU Affero General Public License
C  along with EGSnrc. If not, see <http://www.gnu.org/licenses/>.
C
C##############################################################################
C
C  Author:          Iwan Kawrakow, 2003
C
C  Contributors:
C
C##############################################################################


C******************************************************************************
C
C Print the configuration name as specified suring the configuration
C process to the unit specified by ounit.
C
C*****************************************************************************

      subroutine egs_print_configuration_name(ounit)
      integer ounit
      write(6,'(a,$)') 'linux'
      return
      end

C******************************************************************************
C
C Assign the configuration name as specified suring the configuration
C process to the string pointed to by res
C
C******************************************************************************

      subroutine egs_get_configuration_name(res)
      character*(*) res
      integer l1,l2
      l1 = lnblnk1('linux')
      l2 = len(res)
      res(:l2) = ' '
      if( l2.ge.l1 ) then
        res(:l1) = 'linux'
      else
        res(:l2) = 'linux'
      end if
      return
      end


C##############################################################################
C
C  EGSnrc hostname subroutines v1
C  Copyright (C) 2015 National Research Council Canada
C
C  This file is part of EGSnrc.
C
C  EGSnrc is free software: you can redistribute it and/or modify it under
C  the terms of the GNU Affero General Public License as published by the
C  Free Software Foundation, either version 3 of the License, or (at your
C  option) any later version.
C
C  EGSnrc is distributed in the hope that it will be useful, but WITHOUT ANY
C  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
C  FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for
C  more details.
C
C  You should have received a copy of the GNU Affero General Public License
C  along with EGSnrc. If not, see <http://www.gnu.org/licenses/>.
C
C##############################################################################
C
C  Author:          Iwan Kawrakow, 2003
C
C  Contributors:
C
C##############################################################################


C*****************************************************************************
C
C Print the host name to the unit specified by ounit without inserting
C a new line character.
C
C*****************************************************************************

      subroutine egs_print_hostnm(ounit)
      integer ounit
      character*256 string
      integer res,hostnm,lnblnk1
      res = hostnm(string)
      if( res.ne.0 ) then
        write(6,'(a,a)') 'hostnm returned with a non-zero status '
        stop
      end if
      write(ounit,'(a,$)') string(:lnblnk1(string))
      return
      end

C*****************************************************************************
C
C Assign the host name to the string pointed to be hname.
C
C*****************************************************************************

      subroutine egs_get_hostnm(hname)
      character*(*) hname
      character*256 string
      integer res,hostnm,lnblnk1,l1,l2,l
      res = hostnm(string)
      if( res.ne.0 ) then
        write(6,'(a,a)') 'hostnm returned with a non-zero status '
        stop
      end if
      l1 = lnblnk1(string)
      l2 = len(hname)
      hname(:l2) = ' '
      l = min(l1,l2)
      hname(:l) = string(:l)
      return
      end

      subroutine egs_init
      implicit none
      common/my_times/ t_elapsed, t_cpu, t_first
      real*8 t_elapsed, t_cpu
      integer t_first(8)
      real egs_tot_time,egs_etime
      real*8 dum
      call egs_set_defaults
      call egs_check_arguments
      call egs_init1
      return
      end
      subroutine egs_init1
      implicit none
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      COMMON/MISC/  DUNIT,KMPI,KMPO,RHOR(1),MED(1),IRAYLR(1),IPHOTONUCR(
     *1)
      real*8 DUNIT,  RHOR
      integer*4 KMPI,  KMPO
      integer*2 MED,  IRAYLR,  IPHOTONUCR
      common/my_times/ t_elapsed, t_cpu, t_first
      real*8 t_elapsed, t_cpu
      integer t_first(8)
      real egs_tot_time,egs_etime
      integer l, lnblnk1, l1, l2
      integer i
      character arg*256,tmp_string*512, tmp1_string*512, ucode_dir*512,
     *line*80, line1*80,dattim*24
      logical have_input,egs_isdir,egs_strip_extension,ex, on_egs_home,i
     *s_opened
      integer*4 mypid
      integer getpid
      integer istat, egs_system, u, pos1, pos2,egs_get_unit,itmp
      real*8 dum
      t_elapsed = 0
      t_cpu = egs_etime()
      dum = egs_tot_time(1)
      call egs_date_and_time(t_first)
      DO 2591 i=1,len(line)
        line(i:i) = '='
2591  CONTINUE
2592  CONTINUE
      DO 2601 i=1,len(line1)
        line1(i:i) = '.'
2601  CONTINUE
2602  CONTINUE
      IF ((.NOT.is_pegsless)) THEN
        on_egs_home = .false.
        inquire(file=pegs_file,exist=ex)
        IF (( ex )) THEN
          kmpi=egs_get_unit(kmpi)
          IF ((kmpi.LT.0)) THEN
            write(i_log,'(/a)') '***************** Error: '
            write(i_log,*) 'failed to get a free Fortran I/O unit for pe
     *gs file'
            write(i_log,'(/a)') '***************** Quiting now.'
            call exit(1)
          END IF
          open(kmpi,file=pegs_file,status='old',err=2610)
          goto 2620
        END IF
        arg = pegs_file(:lnblnk1(pegs_file))
        ex = egs_strip_extension(arg,'.pegs4dat')
        l = lnblnk1(egs_home)
        l1 = lnblnk1('pegs4data') + 2*lnblnk1('/')
        l2 = lnblnk1(arg) + lnblnk1('.pegs4dat')
        IF (( l + l1 + l2 .GT. 256 )) THEN
          write(i_log,'(/a)') '***************** Warning: '
          write(i_log,*) 'pegs4 data file name (including absolute path)
     *'
          write(i_log,'(a,i4,a)') 'is too long (',l+l1+l2,') characters'
        ELSE
          pegs_file = egs_home(:lnblnk1(egs_home)) // 'pegs4' // '/' //
     *    'data' // '/' // arg(:lnblnk1(arg)) // '.pegs4dat'
          inquire(file=pegs_file,exist=ex)
          IF (( ex )) THEN
            kmpi=egs_get_unit(kmpi)
            IF ((kmpi.LT.0)) THEN
              write(i_log,'(/a)') '***************** Error: '
              write(i_log,*) 'failed to get a free Fortran I/O unit for
     *pegs file'
              write(i_log,'(/a)') '***************** Quiting now.'
              call exit(1)
            END IF
            open(kmpi,file=pegs_file,status='old',err=2610)
            on_egs_home = .true.
            goto 2620
          END IF
        END IF
        l = lnblnk1(hen_house)
        IF (( l + l1 + l2 .GT. 256 )) THEN
          write(i_log,'(/a)') '***************** Warning: '
          write(i_log,*) 'pegs4 data file name (including absolute path)
     *'
          write(i_log,'(a,i4,a)') 'is too long (',l+l1+l2,') characters'
        ELSE
          pegs_file = hen_house(:lnblnk1(hen_house)) // 'pegs4' // '/' /
     *    / 'data' // '/' // arg(:lnblnk1(arg)) // '.pegs4dat'
          inquire(file=pegs_file,exist=ex)
          IF (( ex )) THEN
            kmpi=egs_get_unit(kmpi)
            IF ((kmpi.LT.0)) THEN
              write(i_log,'(/a)') '***************** Error: '
              write(i_log,*) 'failed to get a free Fortran I/O unit for
     *pegs file'
              write(i_log,'(/a)') '***************** Quiting now.'
              call exit(1)
            END IF
            open(kmpi,file=pegs_file,status='old',err=2610)
            goto 2620
          END IF
        END IF
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'could not find pegs4 file named ',arg(:lnblnk1(a
     *  rg))
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
2620  CONTINUE
      DO 2631 i=1,len(tmp_string)
        tmp_string(i:i) = ' '
2631  CONTINUE
2632  CONTINUE
      tmp_string = hen_house(:lnblnk1(hen_house)) // 'data' // '/'
      i_nist_data=76
      i_incoh=78
      i_photo_relax=77
      i_photo_cs=79
      i_mscat=11
      DO 2641 i=1,len(tmp1_string)
        tmp1_string(i:i) = ' '
2641  CONTINUE
2642  CONTINUE
      tmp1_string = tmp_string(:lnblnk1(tmp_string)) // 'photo_cs.data'
      inquire(file=tmp1_string,exist=ex,opened=is_opened,number=itmp)
      IF (( .NOT.ex )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'EGSnrc data file ','photo_cs.data',' does not ex
     *ist'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      IF (( .NOT.is_opened )) THEN
        i_photo_cs=egs_get_unit(i_photo_cs)
        IF ((i_photo_cs.LT.0)) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,*) 'failed to get a free Fortran I/O unit for data
     * file ', tmp1_string(:lnblnk1(tmp1_string))
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        open(i_photo_cs,file=tmp1_string,status='old',err=2650)
      ELSE
        i_photo_cs = itmp
      END IF
      DO 2661 i=1,len(tmp1_string)
        tmp1_string(i:i) = ' '
2661  CONTINUE
2662  CONTINUE
      tmp1_string = tmp_string(:lnblnk1(tmp_string)) // 'msnew.data'
      inquire(file=tmp1_string,exist=ex,opened=is_opened,number=itmp)
      IF (( .NOT.ex )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'EGSnrc data file ','msnew.data',' does not exist
     *'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      IF (( .NOT.is_opened )) THEN
        i_mscat=egs_get_unit(i_mscat)
        IF ((i_mscat.LT.0)) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,*) 'failed to get a free Fortran I/O unit for data
     * file ', tmp1_string(:lnblnk1(tmp1_string))
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        open(i_mscat,file=tmp1_string,status='old',err=2650)
      ELSE
        i_mscat = itmp
      END IF
      DO 2671 i=1,len(tmp1_string)
        tmp1_string(i:i) = ' '
2671  CONTINUE
2672  CONTINUE
      tmp1_string = tmp_string(:lnblnk1(tmp_string)) // 'incoh.data'
      inquire(file=tmp1_string,exist=ex,opened=is_opened,number=itmp)
      IF (( .NOT.ex )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'EGSnrc data file ','incoh.data',' does not exist
     *'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      IF (( .NOT.is_opened )) THEN
        i_incoh=egs_get_unit(i_incoh)
        IF ((i_incoh.LT.0)) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,*) 'failed to get a free Fortran I/O unit for data
     * file ', tmp1_string(:lnblnk1(tmp1_string))
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        open(i_incoh,file=tmp1_string,status='old',err=2650)
      ELSE
        i_incoh = itmp
      END IF
      DO 2681 i=1,len(tmp1_string)
        tmp1_string(i:i) = ' '
2681  CONTINUE
2682  CONTINUE
      tmp1_string = tmp_string(:lnblnk1(tmp_string)) // 'photo_relax.dat
     *a'
      inquire(file=tmp1_string,exist=ex,opened=is_opened,number=itmp)
      IF (( .NOT.ex )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'EGSnrc data file ','photo_relax.data',' does not
     * exist'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      IF (( .NOT.is_opened )) THEN
        i_photo_relax=egs_get_unit(i_photo_relax)
        IF ((i_photo_relax.LT.0)) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,*) 'failed to get a free Fortran I/O unit for data
     * file ', tmp1_string(:lnblnk1(tmp1_string))
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        open(i_photo_relax,file=tmp1_string,status='old',err=2650)
      ELSE
        i_photo_relax = itmp
      END IF
      DO 2691 i=1,len(ucode_dir)
        ucode_dir(i:i) = ' '
2691  CONTINUE
2692  CONTINUE
      ucode_dir = egs_home(:lnblnk1(egs_home)) // user_code(:lnblnk1(use
     *r_code)) // '/'
      have_input = .false.
      i_input=5
      IF (( lnblnk1(input_file) .GT. 0 )) THEN
        have_input = .true.
        l = lnblnk1(egs_home)
        l1 = lnblnk1(user_code)+1
        l2 = lnblnk1(input_file) + lnblnk1('.egsinp')
        IF (( l + l1 + l2 .GT. 1024 )) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,*) 'input file name (including path) is too long '
     *    ,l+l1+l2
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        ex = egs_strip_extension(input_file,'.egsinp')
        tmp_string = ucode_dir(:lnblnk1(ucode_dir)) // input_file(:lnbln
     *  k1(input_file)) // '.egsinp'
        inquire(file=tmp_string,exist=ex)
        IF (( .NOT.ex )) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,*) 'Input file ',tmp_string(:lnblnk1(tmp_string)),
     *    ' does not exist.'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        open(i_input,file=tmp_string,status='old',err=2700)
      END IF
      DO 2711 i=1,len(work_dir)
        work_dir(i:i) = ' '
2711  CONTINUE
2712  CONTINUE
      work_dir = 'egsrun_'
      mypid = getpid()
      call egs_itostring(work_dir,mypid,.false.)
      call egs_get_hostnm(host_name)
      IF((lnblnk1(host_name) .LT. 1))host_name = 'unknown'
      IF (( have_input )) THEN
        work_dir = work_dir(:lnblnk1(work_dir)) // '_' // input_file(:ln
     *  blnk1(input_file)) // '_' // host_name(:lnblnk1(host_name)) // '
     */'
      ELSE
        work_dir = work_dir(:lnblnk1(work_dir)) // '_noinput_' // host_n
     *  ame(:lnblnk1(host_name)) // '/'
      END IF
      DO 2721 i=1,len(tmp_string)
        tmp_string(i:i) = ' '
2721  CONTINUE
2722  CONTINUE
      tmp_string = ucode_dir(:lnblnk1(ucode_dir)) // work_dir(:lnblnk1(w
     *ork_dir))
      DO 2731 i=1,lnblnk1(tmp_string)
        IF (( tmp_string(i:i) .EQ. '/' )) THEN
          tmp_string(i:i) = '/'
        END IF
2731  CONTINUE
2732  CONTINUE
      ex = egs_isdir(tmp_string)
      IF (( ex )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'a directory named ',tmp_string(:lnblnk1(tmp_stri
     *  ng)),' already exists?'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      tmp1_string = 'mkdir ' // tmp_string(:lnblnk1(tmp_string))
      l = lnblnk1(tmp1_string)
      tmp1_string(l+1:l+1) = char(0)
      istat = egs_system(tmp1_string)
      IF (( istat .NE. 0 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'failed to create working directory ',tmp1_string
     *  (:lnblnk1(tmp1_string))
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      call egs_open_units(.true.)
      write(i_log,'(a)') line
      write(i_log,'(a,a,t55,a,$)') 'EGSnrc version 4 for ','x86_64-unkno
     *wn-linux-gnu',' '
      call egs_get_fdate(dattim)
      write(i_log,'(a,/,a)') dattim,line
      pos1 = lnblnk1('output file(s)')
      pos2 = 80 - lnblnk1('linux')
      pos2 = min(pos2,80-lnblnk1(user_code))
      DO 2741 i=1,len(tmp_string)
        tmp_string(i:i) = ' '
2741  CONTINUE
2742  CONTINUE
      tmp_string = pegs_file
      call egs_strip_path(tmp_string)
      ex = egs_strip_extension(tmp_string,'.pegs4dat')
      IF (( on_egs_home )) THEN
        tmp_string = tmp_string(:lnblnk1(tmp_string)) // ' on EGS_HOME'
      ELSE
        tmp_string = tmp_string(:lnblnk1(tmp_string)) // ' on HEN_HOUSE'
      END IF
      IF (( lnblnk1(tmp_string) .GT. lnblnk1(pegs_file) )) THEN
        DO 2751 i=1,len(tmp_string)
          tmp_string(i:i) = ' '
2751    CONTINUE
2752    CONTINUE
        tmp_string = pegs_file
      END IF
      pos2 = min(pos2,80-lnblnk1(tmp_string))
      pos2 = min(pos2,80-lnblnk1(host_name))
      IF((have_input))pos2 = min(pos2,80-lnblnk1(input_file))
      pos2 = min(pos2,80-lnblnk1(output_file))
      IF((pos2 .LT. pos1+2))pos2 = pos1 + 2
      write(i_log,'(a,$)') 'configuration'
      l = pos2 - lnblnk1('configuration')
      write(i_log,'(a,$)') line1(:l)
      write(i_log,'(a)') 'linux'
      write(i_log,'(a,$)') 'user code'
      l = pos2 - lnblnk1('user code')
      write(i_log,'(a,$)') line1(:l)
      write(i_log,'(a)') user_code(:lnblnk1(user_code))
      write(i_log,'(a,$)') 'pegs file'
      l = pos2 - lnblnk1('pegs file')
      write(i_log,'(a,$)') line1(:l)
      write(i_log,'(a)') tmp_string(:lnblnk1(tmp_string))
      write(i_log,'(a,$)') 'using host'
      l = pos2 - lnblnk1('using host')
      write(i_log,'(a,$)') line1(:l)
      write(i_log,'(a)') host_name(:lnblnk1(host_name))
      IF (( have_input )) THEN
        write(i_log,'(a,$)') 'input file'
        l = pos2 - lnblnk1('input file')
        write(i_log,'(a,$)') line1(:l)
        write(i_log,'(a)') input_file(:lnblnk1(input_file))
      END IF
      write(i_log,'(a,$)') 'output file(s)'
      l = pos2 - lnblnk1('output file(s)')
      write(i_log,'(a,$)') line1(:l)
      write(i_log,'(a)') output_file(:lnblnk1(output_file))
      IF (( n_parallel .GT. 0 )) THEN
        write(i_log,'(a,$)') 'number of parallel jobs'
        l = pos2 - lnblnk1('number of parallel jobs')
        write(i_log,'(a,$)') line1(:l)
        write(i_log,'(i2)') n_parallel
        write(i_log,'(a,$)') 'job number'
        l = pos2 - lnblnk1('job number')
        write(i_log,'(a,$)') line1(:l)
        write(i_log,'(i2)') i_parallel
      END IF
      write(i_log,'(a)') line
      return
2700  write(i_log,'(/a)') '***************** Error: '
      write(i_log,*) 'failed to open input file ',tmp_string(:lnblnk1(tm
     *p_string))
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
2610  write(i_log,'(/a)') '***************** Error: '
      write(i_log,*) 'failed to open existing pegs file ',pegs_file(:lnb
     *lnk1(pegs_file))
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
2650  write(i_log,'(/a)') '***************** Error: '
      write(i_log,*) 'failed to open EGSnrc data file ',tmp1_string(:lnb
     *lnk1(tmp1_string))
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
      return
      end
      subroutine egs_check_arguments
      implicit none
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      character arg*256,tmp_string*512, line1*80
      logical have_arg,egs_isdir,egs_strip_extension,ex, on_egs_home
      integer narg, iargc, i, lnblnk1, l, l2,i_help,egs_get_unit
      narg = iargc()
      IF((narg .LT. 1))return
      have_arg = .false.
      DO 2761 i=1,narg-1
        call getarg(i,arg)
        l = lnblnk1(arg)
        IF (( ( l .EQ. lnblnk1('-H') .AND. arg(:l) .EQ. '-H' ) .OR. ( l
     *  .EQ. lnblnk1('--hen-house') .AND. arg(:l) .EQ. '--hen-house' ) )
     *  ) THEN
          have_arg = .true.
          call getarg(i+1,arg)
          GO TO2762
        END IF
2761  CONTINUE
2762  CONTINUE
      IF (( have_arg )) THEN
        l = lnblnk1(arg)
        DO 2771 i=1,len(hen_house)
          hen_house(i:i) = ' '
2771    CONTINUE
2772    CONTINUE
        IF (( l .GT. 0 )) THEN
          IF (( l .GT. 254 )) THEN
            write(i_log,'(/a)') '***************** Error: '
            write(i_log,'(a,i5)') ' HEN_HOUSE argument is too long',l
            write(i_log,'(/a)') '***************** Quiting now.'
            call exit(1)
          END IF
          hen_house(:l) = arg(:lnblnk1(arg))
          IF((hen_house(l:l) .NE. '/'))hen_house(l+1:l+1) = '/'
        ELSE
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,'(a)') ' empty argument after -H'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        DO 2781 i=1,lnblnk1(hen_house)
          IF (( hen_house(i:i) .EQ. '/' )) THEN
            hen_house(i:i) = '/'
          END IF
2781    CONTINUE
2782    CONTINUE
      END IF
      IF (( .NOT.egs_isdir(hen_house) )) THEN
        write(i_log,'(/a)') '***************** Warning: '
        write(i_log,'(a,a)') ' HEN_HOUSE directory ',hen_house(:lnblnk1(
     *  hen_house))
        write(i_log,'(a)') 'does not exist. Hope you know what you are d
     *oing.'
      END IF
      have_arg = .false.
      DO 2791 i=1,narg
        call getarg(i,arg)
        l = lnblnk1(arg)
        IF (( ( l .EQ. lnblnk1('-h') .AND. arg(:l) .EQ. '-h' ) .OR. ( l
     *  .EQ. lnblnk1('--help') .AND. arg(:l) .EQ. '--help' ) )) THEN
          have_arg = .true.
          GO TO2792
        END IF
2791  CONTINUE
2792  CONTINUE
      IF (( have_arg )) THEN
        call getarg(0,arg)
        call egs_strip_path(arg)
        write(i_log,'(//,a,a,a,//)') 'Usage: ',arg(:lnblnk1(arg)),' [arg
     *s] '
        tmp_string = hen_house(:lnblnk1(hen_house)) // 'pieces/help_mess
     *age'
        i_help=98
        i_help=egs_get_unit(i_help)
        IF ((i_help.LT.0)) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,*) 'failed to get a free Fortran I/O unit for help
     * file'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        open(i_help,file=tmp_string,status='old',err=2800)
2811    CONTINUE
          read(i_help,'(a)',err=2820,end=2820) line1
          write(i_log,'(a)') line1
        GO TO 2811
2812    CONTINUE
2820    CONTINUE
        call exit(0)
2800    CONTINUE
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'Did not find the help_message file!'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      have_arg = .false.
      DO 2831 i=1,narg
        call getarg(i,arg)
        l = lnblnk1(arg)
        IF (( ( l .EQ. lnblnk1('-b') .AND. arg(:l) .EQ. '-b' ) .OR. ( l
     *  .EQ. lnblnk1('--batch') .AND. arg(:l) .EQ. '--batch' ) )) THEN
          have_arg = .true.
          GO TO2832
        END IF
2831  CONTINUE
2832  CONTINUE
      IF((have_arg))is_batch = .true.
      have_arg = .false.
      DO 2841 i=1,narg-1
        call getarg(i,arg)
        l = lnblnk1(arg)
        IF (( ( l .EQ. lnblnk1('-P') .AND. arg(:l) .EQ. '-P' ) .OR. ( l
     *  .EQ. lnblnk1('--parallel') .AND. arg(:l) .EQ. '--parallel' ) ))
     *  THEN
          have_arg = .true.
          call getarg(i+1,arg)
          GO TO2842
        END IF
2841  CONTINUE
2842  CONTINUE
      IF (( have_arg )) THEN
        read(arg,*,err=2850) n_parallel
        IF((n_parallel .LT. 0))goto 2850
        goto 2860
2850    CONTINUE
        write(i_log,'(/a)') '***************** Warning: '
        write(i_log,*) ' Wrong/missing parallel job number argument, -P
     *option ignored'
        n_parallel = 0
2860    CONTINUE
      END IF
      have_arg = .false.
      DO 2871 i=1,narg-1
        call getarg(i,arg)
        l = lnblnk1(arg)
        IF (( ( l .EQ. lnblnk1('-j') .AND. arg(:l) .EQ. '-j' ) .OR. ( l
     *  .EQ. lnblnk1('--job') .AND. arg(:l) .EQ. '--job' ) )) THEN
          have_arg = .true.
          call getarg(i+1,arg)
          GO TO2872
        END IF
2871  CONTINUE
2872  CONTINUE
      IF (( have_arg )) THEN
        read(arg,*,err=2880) i_parallel
        IF((i_parallel .LT. 0))goto 2880
        goto 2890
2880    CONTINUE
        write(i_log,'(/a)') '***************** Warning: '
        write(i_log,*) ' Wrong/missing job argument, -j option ognored'
        i_parallel = 0
2890    CONTINUE
      END IF
      have_arg = .false.
      DO 2901 i=1,narg-1
        call getarg(i,arg)
        l = lnblnk1(arg)
        IF (( ( l .EQ. lnblnk1('-f') .AND. arg(:l) .EQ. '-f' ) .OR. ( l
     *  .EQ. lnblnk1('--first-job') .AND. arg(:l) .EQ. '--first-job' ) )
     *  ) THEN
          have_arg = .true.
          call getarg(i+1,arg)
          GO TO2902
        END IF
2901  CONTINUE
2902  CONTINUE
      IF (( have_arg )) THEN
        read(arg,*,err=2910) first_parallel
        IF((first_parallel .LT. 1))goto 2910
        goto 2920
2910    CONTINUE
        write(i_log,'(/a)') '***************** Warning: '
        write(i_log,*) ' Wrong/missing first job argument, -f option ogn
     *ored'
        first_parallel = 1
2920    CONTINUE
      END IF
      IF (( n_parallel .GT. 0 .OR. i_parallel .GT. 0 )) THEN
        IF (( n_parallel*i_parallel .EQ. 0 )) THEN
          write(i_log,'(/a)') '***************** Warning: '
          write(i_log,*) 'You need to specify number of jobs AND job num
     *ber ', '=> will not use parallel run '
          n_parallel = 0
          i_parallel = 0
        END IF
        IF (( first_parallel .GT. i_parallel )) THEN
          write(i_log,'(/a)') '***************** Warning: '
          write(i_log,*) 'i_parallel (',i_parallel, ') can not be smalle
     *r than first_parallel (',first_parallel,')'
          first_parallel = i_parallel
        END IF
      END IF
      have_arg = .false.
      DO 2931 i=1,narg-1
        call getarg(i,arg)
        l = lnblnk1(arg)
        IF (( ( l .EQ. lnblnk1('-e') .AND. arg(:l) .EQ. '-e' ) .OR. ( l
     *  .EQ. lnblnk1('--egs-home') .AND. arg(:l) .EQ. '--egs-home' ) ))
     *  THEN
          have_arg = .true.
          call getarg(i+1,arg)
          GO TO2932
        END IF
2931  CONTINUE
2932  CONTINUE
      IF (( have_arg )) THEN
        l = lnblnk1(arg)
        DO 2941 i=1,len(egs_home)
          egs_home(i:i) = ' '
2941    CONTINUE
2942    CONTINUE
        IF (( l .EQ. 0 )) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,'(a)') ' empty argument after -e'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        IF (( l .GT. 254 )) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,'(a,i5)') ' EGS_HOME argument is too long ',l
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        egs_home(:l) = arg(:lnblnk1(arg))
        IF((egs_home(l:l) .NE. '/'))egs_home(l+1:l+1) = '/'
        DO 2951 i=1,lnblnk1(egs_home)
          IF (( egs_home(i:i) .EQ. '/' )) THEN
            egs_home(i:i) = '/'
          END IF
2951    CONTINUE
2952    CONTINUE
      END IF
      IF (( .NOT.egs_isdir(egs_home) )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) ' EGS_HOME directory ',egs_home(:lnblnk1(egs_home
     *  )),' does not exist.'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      on_egs_home = .false.
      is_pegsless=.false.
      have_arg = .false.
      DO 2961 i=1,narg-1
        call getarg(i,arg)
        l = lnblnk1(arg)
        IF (( ( l .EQ. lnblnk1('-p') .AND. arg(:l) .EQ. '-p' ) .OR. ( l
     *  .EQ. lnblnk1('--pegs-file') .AND. arg(:l) .EQ. '--pegs-file' ) )
     *  ) THEN
          have_arg = .true.
          call getarg(i+1,arg)
          GO TO2962
        END IF
2961  CONTINUE
2962  CONTINUE
      IF (( .NOT.have_arg )) THEN
        write(i_log,'(/a)') '***************** Warning: '
        write(i_log,*) 'No pegs4 file name supplied.  Will assume you ar
     *e running    in pegs-less mode with media details specified in inp
     *ut file.'
        is_pegsless=.true.
      ELSE
        pegs_file = arg(:lnblnk1(arg))
      END IF
      call egs_get_usercode(user_code)
      have_arg = .false.
      DO 2971 i=1,narg-1
        call getarg(i,arg)
        l = lnblnk1(arg)
        IF (( ( l .EQ. lnblnk1('-i') .AND. arg(:l) .EQ. '-i' ) .OR. ( l
     *  .EQ. lnblnk1('--input') .AND. arg(:l) .EQ. '--input' ) )) THEN
          have_arg = .true.
          call getarg(i+1,arg)
          GO TO2972
        END IF
2971  CONTINUE
2972  CONTINUE
      IF (( have_arg )) THEN
        ex = egs_strip_extension(arg,'.egsinp')
        l2 = lnblnk1(arg) + lnblnk1('.egsinp')
        IF (( l2 .GT. 256 )) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,*) 'input file name is too long ',l2
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        input_file = arg(:lnblnk1(arg))
      END IF
      have_arg = .false.
      DO 2981 i=1,narg-1
        call getarg(i,arg)
        l = lnblnk1(arg)
        IF (( ( l .EQ. lnblnk1('-o') .AND. arg(:l) .EQ. '-o' ) .OR. ( l
     *  .EQ. lnblnk1('--output') .AND. arg(:l) .EQ. '--output' ) )) THEN
          have_arg = .true.
          call getarg(i+1,arg)
          GO TO2982
        END IF
2981  CONTINUE
2982  CONTINUE
      IF (( have_arg )) THEN
        l = lnblnk1(arg)
        IF (( l .GT. 256 )) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,*) 'output file name is too long ',l
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        output_file(:l) = arg(:lnblnk1(arg))
      ELSE
        IF (( lnblnk1(input_file) .GT. 0 )) THEN
          output_file(:lnblnk1(input_file)) = input_file(:lnblnk1(input_
     *    file))
        ELSE
          output_file = 'test'
        END IF
      END IF
      return
      end
      subroutine egs_open_units(flag)
      implicit none
      logical flag
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      character tmp_string*1024, tmp1_string*1024, tmp2_string*1024, uco
     *de_dir*1024, input_line*100, arg*20
      integer i,lnblnk1,u,l,istart,egs_get_unit,i_iofile
      logical ex,is_open
      DO 2991 i=1,len(tmp_string)
        tmp_string(i:i) = ' '
2991  CONTINUE
2992  CONTINUE
      DO 3001 i=1,len(ucode_dir)
        ucode_dir(i:i) = ' '
3001  CONTINUE
3002  CONTINUE
      ucode_dir = egs_home(:lnblnk1(egs_home)) // user_code(:lnblnk1(use
     *r_code)) // '/'
      IF (( flag )) THEN
        tmp_string = ucode_dir(:lnblnk1(ucode_dir)) // work_dir(:lnblnk1
     *  (work_dir))
      ELSE
        tmp_string = ucode_dir(:lnblnk1(ucode_dir))
      END IF
      tmp_string = tmp_string(:lnblnk1(tmp_string)) // output_file(:lnbl
     *nk1(output_file))
      IF (( i_parallel .GT. 0 )) THEN
        tmp_string = tmp_string(:lnblnk1(tmp_string)) // '_w'
        call egs_itostring(tmp_string,i_parallel,.false.)
      END IF
      DO 3011 i=1,len(tmp1_string)
        tmp1_string(i:i) = ' '
3011  CONTINUE
3012  CONTINUE
      i_log=6
      IF (( is_batch )) THEN
        tmp1_string = tmp_string(:lnblnk1(tmp_string)) // '.egslog'
        open(i_log,file=tmp1_string,status='unknown',err=3020)
      END IF
      DO 3031 i=1,len(tmp2_string)
        tmp2_string(i:i) = ' '
3031  CONTINUE
3032  CONTINUE
      tmp2_string = ucode_dir(:lnblnk1(ucode_dir)) // user_code(:lnblnk1
     *(user_code)) // '.io'
      inquire(file=tmp2_string,exist=ex)
      n_files = 0
      IF (( ex )) THEN
        i_iofile=99
        i_iofile=egs_get_unit(i_iofile)
        IF ((i_iofile.LT.1)) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,*) 'failed to get a free Fortran I/O unit for .io
     *file'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        open(i_iofile,file=tmp2_string,status='old',err=3040)
3051    CONTINUE
          read(i_iofile,'(a)',err=3060,end=3060) input_line
          IF((input_line(1:1) .EQ. '#'))GO TO3051
          read(input_line,*,err=3070,end=3070) u
          istart = 1
          DO 3081 i=lnblnk1(input_line),1,-1
            IF (( input_line(i:i) .EQ. ' ' )) THEN
              istart = i+1
              GO TO3082
            END IF
3081      CONTINUE
3082      CONTINUE
          DO 3091 i=1,len(arg)
            arg(i:i) = ' '
3091      CONTINUE
3092      CONTINUE
          DO 3101 i=istart,lnblnk1(input_line)
            arg(i+1-istart:i+1-istart) = input_line(i:i)
3101      CONTINUE
3102      CONTINUE
          inquire(unit=u,opened=is_open)
          IF (( is_open )) THEN
            write(i_log,'(/a)') '***************** Warning: '
            write(i_log,'(a,i3,a,a,a,/,a,/,a,/)') 'Unit ',u,' which you
     *want to connect to a ', arg(:lnblnk1(arg)),' file ', 'is already i
     *n use. Will assume this code is being used as', 'a shared library
     *source and this file will be opened explicitly.'
          ELSE
            n_files = n_files + 1
            IF (( n_files .GT. 20 )) THEN
              write(i_log,'(/a)') '***************** Error: '
              write(i_log,*) 'Too many units requested in .io.', ' Incre
     *as $mx_units and retry'
              write(i_log,'(/a)') '***************** Quiting now.'
              call exit(1)
            END IF
            file_units(n_files) = u
            DO 3111 i=1,len(file_extensions(n_files))
              file_extensions(n_files)(i:i) = ' '
3111        CONTINUE
3112        CONTINUE
            l = lnblnk1(arg)
            IF (( l .GT. 10 )) THEN
              write(i_log,'(/a)') '***************** Error: '
              write(i_log,*) 'extension ',arg(:lnblnk1(arg)),' is longer
     * than ', 10,' chars. ', 'Increase $max_extension_length and retry
     *'
              write(i_log,'(/a)') '***************** Quiting now.'
              call exit(1)
            END IF
            file_extensions(n_files) = arg(:lnblnk1(arg))
            tmp1_string = tmp_string(:lnblnk1(tmp_string)) // arg(:lnbln
     *      k1(arg))
            open(u,file=tmp1_string,status='unknown')
          END IF
3070      CONTINUE
        GO TO 3051
3052    CONTINUE
3060    close(i_iofile)
      END IF
      return
3020  write(i_log,'(/a)') '***************** Error: '
      write(i_log,*) 'failed to open output file ',tmp1_string(:lnblnk1(
     *tmp1_string))
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
3040  write(i_log,'(/a)') '***************** Error: '
      write(i_log,*) 'failed to open existing .io file',tmp2_string(:lnb
     *lnk1(tmp2_string))
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
      return
      end
      subroutine egs_finish
      implicit none
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      common/my_times/ t_elapsed, t_cpu, t_first
      real*8 t_elapsed, t_cpu
      integer t_first(8)
      real egs_tot_time,egs_etime
      character line*80,base*512,base1*512,tmp_string*512,junk_file*128,
     *fname*512
      character dattim*24
      integer i,l,lnblnk1,istat,egs_system,n_open,unlink,i_junk,egs_get_
     *unit
      logical is_open,egs_isdir
      real*8 t1,t2,tt_cpu
      DO 3121 i=1,len(line)
        line(i:i) = '='
3121  CONTINUE
3122  CONTINUE
      IF (( n_parallel .EQ. 0 .OR. i_parallel .GT. 0 )) THEN
        t_elapsed = egs_tot_time(1)
        tt_cpu = egs_etime() - t_cpu
        t1 = t_elapsed
        t2 = t1/3600
        write(i_log,'(//a,/,a,/)') line,'Finished simulation'
        write(i_log,'(2x,a,t30,f9.1,a,f7.3,a)') 'Elapsed time: ',t1,' s
     *(',t2,' h)'
        t1 = tt_cpu
        t2 = t1/3600
        write(i_log,'(2x,a,t30,f9.1,a,f7.3,a)') 'CPU time:',t1,' s (',t2
     *  ,' h)'
        write(i_log,'(2x,a,t30,f10.3)') 'Ratio:',t_elapsed/tt_cpu
      END IF
      call egs_get_fdate(dattim)
      write(i_log,'(//a,t56,a,/,a)') 'End of run ',dattim,line
      n_open=0
      DO 3131 i=1,len(base)
        base(i:i) = ' '
3131  CONTINUE
3132  CONTINUE
      base = egs_home(:lnblnk1(egs_home)) // user_code(:lnblnk1(user_cod
     *e))
      DO 3141 i=1,99
        IF (( is_batch .OR. i .NE. i_log )) THEN
          inquire(i,opened=is_open)
          IF (( is_open )) THEN
            inquire(i,name=fname)
            IF ((index(fname(:lnblnk1(fname)),base(:lnblnk1(base))).GT.0
     *      )) THEN
              close(i)
              n_open = n_open+1
            END IF
          END IF
        END IF
3141  CONTINUE
3142  CONTINUE
      IF (( lnblnk1(work_dir) .EQ. 0 )) THEN
        return
      END IF
      DO 3151 i=1,len(base)
        base(i:i) = ' '
3151  CONTINUE
3152  CONTINUE
      base = egs_home(:lnblnk1(egs_home)) // user_code(:lnblnk1(user_cod
     *e)) // '/' // work_dir(:lnblnk1(work_dir))
      DO 3161 i=1,lnblnk1(base)
        IF (( base(i:i) .EQ. '/' )) THEN
          base(i:i) = '/'
        END IF
3161  CONTINUE
3162  CONTINUE
      IF (( egs_isdir(base) )) THEN
        DO 3171 i=1,len(tmp_string)
          tmp_string(i:i) = ' '
3171    CONTINUE
3172    CONTINUE
        DO 3181 i=1,len(junk_file)
          junk_file(i:i) = ' '
3181    CONTINUE
3182    CONTINUE
        junk_file = work_dir(:lnblnk1(work_dir))
        l = lnblnk1(junk_file)
        junk_file(l:l) = ' '
        junk_file = junk_file(:lnblnk1(junk_file)) // '_junk'
        tmp_string = base(:lnblnk1(base)) // junk_file(:lnblnk1(junk_fil
     *  e))
        i_junk=99
        i_junk=egs_get_unit(i_junk)
        IF ((i_junk.LT.0)) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,*) 'failed to get a free Fortran I/O unit for junk
     * file'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        open(i_junk,file=tmp_string,status='unknown')
        write(i_junk,*) 'junk'
        close(i_junk)
        DO 3191 i=1,len(base1)
          base1(i:i) = ' '
3191    CONTINUE
3192    CONTINUE
        base = egs_home(:lnblnk1(egs_home)) // user_code(:lnblnk1(user_c
     *  ode)) // '/' // work_dir(:lnblnk1(work_dir))
        base1 = egs_home(:lnblnk1(egs_home)) // user_code(:lnblnk1(user_
     *  code))
        DO 3201 i=1,len(tmp_string)
          tmp_string(i:i) = ' '
3201    CONTINUE
3202    CONTINUE
        tmp_string = 'mv -f ' // base(:lnblnk1(base)) // '*  ' // base1(
     *  :lnblnk1(base1))
        l = lnblnk1(tmp_string)+1
        tmp_string(l:l) = char(0)
        istat = egs_system(tmp_string)
        IF (( istat .NE. 0 )) THEN
          write(i_log,'(/a)') '***************** Warning: '
          write(i_log,*) 'Moving files from working directory failed ?'
          write(i_log,*) '=> will not remove working directory'
        ELSE
          DO 3211 i=1,len(tmp_string)
            tmp_string(i:i) = ' '
3211      CONTINUE
3212      CONTINUE
          tmp_string = 'rm -rf ' // base(:lnblnk1(base))
          l = lnblnk1(tmp_string)+1
          tmp_string(l:l) = char(0)
          istat = egs_system(tmp_string)
          IF (( istat .NE. 0 )) THEN
            write(i_log,'(/a)') '***************** Warning: '
            write(i_log,*) 'Failed to remove working directory ', work_d
     *      ir(:lnblnk1(work_dir))
          END IF
          DO 3221 i=1,len(tmp_string)
            tmp_string(i:i) = ' '
3221      CONTINUE
3222      CONTINUE
          tmp_string = base1(:lnblnk1(base1)) // '/' // junk_file(:lnbln
     *    k1(junk_file))
          l = lnblnk1(tmp_string)+1
          tmp_string(l:l) = char(0)
          istat = unlink(tmp_string)
        END IF
      END IF
      DO 3231 i=1,len(work_dir)
        work_dir(i:i) = ' '
3231  CONTINUE
3232  CONTINUE
      return
      end
      subroutine egs_set_defaults
      implicit none
      COMMON/BOUNDS/ECUT(1),PCUT(1),VACDST
      real*8 ECUT,  PCUT,  VACDST
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      common/compton_data/ iz_array(1538),  be_array(1538),  Jo_array(15
     *38),  erfJo_array(1538),   ne_array(1538),  shn_array(1538),
     *shell_array(200,1), eno_array(200,1), eno_atbin_array(200,1), n_sh
     *ell(1), radc_flag,  ibcmp(1)
      integer*4 iz_array,ne_array,shn_array,eno_atbin_array, shell_array
     *,n_shell,radc_flag
      real*8 be_array,Jo_array,erfJo_array,eno_array
      integer*2 ibcmp
      COMMON/EDGE/binding_energies(30,100), interaction_prob(6,100), rel
     *axation_prob(39,100), edge_energies(16,100), edge_number(100), edg
     *e_a(16,100), edge_b(16,100), edge_c(16,100), edge_d(16,100), IEDGF
     *L(1),IPHTER(1)
      real*8 binding_energies,  interaction_prob,    relaxation_prob,  e
     *dge_energies,  edge_a,edge_b,edge_c,edge_d
      integer*2 IEDGFL,  IPHTER
      integer*4 edge_number
      COMMON/ELECIN/ esig_e(1),psig_e(1), esige_max, psige_max, range_ep
     *(0:1,500,1), E_array(500,1), etae_ms0(500,1),etae_ms1(500,1),etap_
     *ms0(500,1),etap_ms1(500,1),q1ce_ms0(500,1),q1ce_ms1(500,1),q1cp_ms
     *0(500,1),q1cp_ms1(500,1),q2ce_ms0(500,1),q2ce_ms1(500,1),q2cp_ms0(
     *500,1),q2cp_ms1(500,1),blcce0(500,1),blcce1(500,1), EKE0(1),EKE1(1
     *), XR0(1),TEFF0(1),BLCC(1),XCC(1), ESIG0(500,1),ESIG1(500,1),PSIG0
     *(500,1),PSIG1(500,1),EDEDX0(500,1),EDEDX1(500,1),PDEDX0(500,1),PDE
     *DX1(500,1),EBR10(500,1),EBR11(500,1),PBR10(500,1),PBR11(500,1),PBR
     *20(500,1),PBR21(500,1),TMXS0(500,1),TMXS1(500,1), expeke1(1), IUNR
     *ST(1),EPSTFL(1),IAPRIM(1), sig_ismonotone(0:1,1)
      real*8 esig_e,   psig_e,   esige_max,  psige_max,  range_ep,  E_ar
     *ray,  etae_ms0,etae_ms1,  etap_ms0,etap_ms1,  q1ce_ms0,q1ce_ms1,
     *q1cp_ms0,q1cp_ms1,  q2ce_ms0,q2ce_ms1,  q2cp_ms0,q2cp_ms1,  blcce0
     *,blcce1,   expeke1,  EKE0,EKE1, XR0,  TEFF0,  BLCC,  XCC,  ESIG0,E
     *SIG1,  PSIG0,PSIG1,  EDEDX0,EDEDX1,  PDEDX0,PDEDX1,  EBR10,EBR11,
     * PBR10,PBR11,  PBR20,PBR21,  TMXS0,TMXS1
      integer*4 IUNRST,  EPSTFL,  IAPRIM
      logical sig_ismonotone
      COMMON/EPCONT/EDEP,EDEP_LOCAL,TSTEP,TUSTEP,USTEP,TVSTEP,VSTEP, RHO
     *F,EOLD,ENEW,EKE,ELKE,GLE,E_RANGE, x_final,y_final,z_final, u_final
     *,v_final,w_final, IDISC,IROLD,IRNEW,IAUSFL(35)
      DOUBLE PRECISION EDEP,  EDEP_LOCAL
      real*8 TSTEP,  TUSTEP,  USTEP,  VSTEP,  TVSTEP,  RHOF,  EOLD,  ENE
     *W,  EKE,  ELKE,  GLE,  E_RANGE, x_final,y_final,z_final,  u_final,
     *v_final,w_final
      integer*4 IDISC,  IROLD,  IRNEW,  IAUSFL
      common/CH_steps/ count_pII_steps,count_all_steps,is_ch_step
      real*8 count_pII_steps,count_all_steps
      logical is_ch_step
      common/ET_control/ smaxir(1),estepe,ximax,  skindepth_for_bca,tran
     *sport_algorithm, bca_algorithm,exact_bca,spin_effects
      real*8 smaxir,  estepe,  ximax,      skindepth_for_bca
      integer*4 transport_algorithm, bca_algorithm
      logical exact_bca,  spin_effects
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/MISC/  DUNIT,KMPI,KMPO,RHOR(1),MED(1),IRAYLR(1),IPHOTONUCR(
     *1)
      real*8 DUNIT,  RHOR
      integer*4 KMPI,  KMPO
      integer*2 MED,  IRAYLR,  IPHOTONUCR
      COMMON/PHOTIN/ EBINDA(1), GE0(1),GE1(1), GMFP0(2000,1),GMFP1(2000,
     *1),GBR10(2000,1),GBR11(2000,1),GBR20(2000,1),GBR21(2000,1), RCO0(1
     *),RCO1(1), RSCT0(100,1),RSCT1(100,1), COHE0(2000,1),COHE1(2000,1),
     *  PHOTONUC0(2000,1),PHOTONUC1(2000,1), DPMFP, MPGEM(1,1), NGR(1)
      real*8 EBINDA,  GE0,GE1,  GMFP0,GMFP1,  GBR10,GBR11,  GBR20,GBR21,
     *  RCO0,RCO1,  RSCT0,RSCT1,  COHE0,COHE1,   PHOTONUC0,PHOTONUC1,  D
     *PMFP
      integer*4
     *                  MPGEM,
     *                          NGR
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/UPHIIN/SINC0,SINC1,SIN0(1002),SIN1(1002)
      real*8 SINC0,SINC1,SIN0,SIN1
      COMMON/UPHIOT/THETA,SINTHE,COSTHE,SINPHI, COSPHI,PI,TWOPI,PI5D2
      real*8 THETA,  SINTHE,  COSTHE,  SINPHI,  COSPHI,  PI,TWOPI,PI5D2
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common/egs_vr/ e_max_rr(1),  prob_RR,  nbr_split,  i_play_RR,
     * i_survived_RR,
     *     n_RR_warning,                                        i_do_rr(
     *1)
      real*8          e_max_rr,prob_RR
      integer*4       nbr_split,i_play_RR,i_survived_RR,n_RR_warning
      integer*2     i_do_rr
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      common/spin_data/ spin_rej(1,0:1,0: 31,0:15,0:31), espin_min,espin
     *_max,espml,b2spin_min,b2spin_max, dbeta2,dbeta2i,dlener,dleneri,dq
     *q1,dqq1i, fool_intel_optimizer
      real*4 spin_rej,espin_min,espin_max,espml,b2spin_min,b2spin_max, d
     *beta2,dbeta2i,dlener,dleneri,dqq1,dqq1i
      logical fool_intel_optimizer
      common/eii_data/ eii_xsection_a( 10000),  eii_xsection_b( 10000),
     * eii_cons(1), eii_a(40),  eii_b(40),  eii_L_factor,  eii_z(40),  e
     *ii_sh(40),  eii_nshells(100),  eii_nsh(1),  eii_first(1,50),  eii_
     *no(1,50),  eii_flag
      real*8 eii_xsection_a,eii_xsection_b,eii_a,eii_b,eii_cons,eii_L_fa
     *ctor
      integer*4 eii_z,eii_sh,eii_nshells
      integer*4 eii_first,eii_no
      integer*4 eii_elements,eii_flag,eii_nsh
      COMMON/rayleigh_inputs/iray_ff_media(1),iray_ff_file(1)
      character*24 iray_ff_media
      character*128 iray_ff_file
      common/emf_inputs/ExIN,EyIN,EzIN,  EMLMTIN,  BxIN, ByIN, BzIN,  Bx
     *, By, Bz,  Bx_new, By_new, Bz_new,  emfield_on
      real*8 ExIN,EyIN,EzIN, EMLMTIN, BxIN,ByIN,BzIN, Bx,By,Bz, Bx_new,B
     *y_new,Bz_new
      logical emfield_on
      common/x_options/eadl_relax,  mcdf_pe_xsections
      logical eadl_relax, mcdf_pe_xsections
      integer i,j,lnblnk1
      CHARACTER*4 MEDIA1(24)
      EQUIVALENCE(MEDIA1(1),MEDIA(1,1))
      character fool_dec
      data MEDIA1/'N','A','I',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','
     *',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '/
      data fool_dec/'/'/
      data fool_intel_optimizer/.false./
      vacdst = 1e8
      DO 3241 i=1,1
        ecut(i) = 0.
        pcut(i) = 0.
        ibcmp(i) = 3
        iedgfl(i) = 1
        iphter(i) = 1
        smaxir(i) = 1e10
        i_do_rr(i) = 0
        e_max_rr(i) = 0
        med(i) = 1
        rhor(i) = 0
        iraylr(i) = 1
        iphotonucr(i) = 0
3241  CONTINUE
3242  CONTINUE
      eii_flag = 0
      eii_xfile = 'Off'
      eii_L_factor = 1.0
      xsec_out = 0
      photon_xsections = 'xcom'
      comp_xsections = 'default'
      eadl_relax = .true.
      mcdf_pe_xsections = .false.
      photonuc_xsections = 'default'
      ExIN=0
      EyIN=0
      EzIN=0
      BxIN=0
      ByIN=0
      BzIN=0
      EMLMTIN=0.02
      Bx=BxIN
      By=ByIN
      Bz=BzIN
      Bx_new=Bx
      By_new=By
      Bz_new=Bz
      emfield_on=.false.
      IF (( ExIN**2+EyIN**2+EzIN**2 + BxIN**2+ByIN**2+BzIN**2 .GT. 0 ))
     *THEN
        emfield_on=.true.
      END IF
      DO 3251 i=1,1
        iraylm(i) = 0
        DO 3261 j=1,len(iray_ff_file(i))
          iray_ff_file(i)(j:j) = ' '
3261    CONTINUE
3262    CONTINUE
        DO 3271 j=1,len(iray_ff_media(i))
          iray_ff_media(i)(j:j) = ' '
3271    CONTINUE
3272    CONTINUE
        ae(i)=0
        ap(i)=0
        ue(i)=0
        up(i)=0
        te(i)=0
        thmoll(i)=0
3251  CONTINUE
3252  CONTINUE
      DO 3281 i=1,30
        DO 3291 j=1,100
          binding_energies(i,j) = 0
3291    CONTINUE
3292    CONTINUE
3281  CONTINUE
3282  CONTINUE
      ibrdst = 1
      ibr_nist = 0
      pair_nrc = 0
      itriplet = 0
      iprdst = 1
      rhof = 1
      DO 3301 i=1,5
        iausfl(i) = 1
3301  CONTINUE
3302  CONTINUE
      DO 3311 i=6,35
        iausfl(i) = 0
3311  CONTINUE
3312  CONTINUE
      ximax = 0.5
      estepe = 0.25
      skindepth_for_bca = 3
      transport_algorithm = 0
      bca_algorithm = 0
      exact_bca = .true.
      spin_effects = .true.
      count_pII_steps = 0
      count_all_steps = 0
      radc_flag = 0
      nmed = 1
      kmpi = 12
      kmpo = 8
      dunit = 1
      rng_seed = 999999
      latchi = 0
      rmt2 = 2*rm
      rmsq = rm*rm
      pi = 4*datan(1d0)
      twopi = 2*pi
      pi5d2 = 2.5*pi
      nbr_split = 1
      i_play_RR = 0
      i_survived_RR = 0
      prob_RR = -1
      n_RR_warning = 0
      DO 3321 i=1,len(hen_house)
        hen_house(i:i) = ' '
3321  CONTINUE
3322  CONTINUE
      i = lnblnk1('/home/user/school/res/EGSnrc/HEN_HOUSE/')
      hen_house(:i) = '/home/user/school/res/EGSnrc/HEN_HOUSE/'
      IF (( '/' .NE. fool_dec )) THEN
        DO 3331 j=1,i
          IF((hen_house(j:j) .EQ. '/'))hen_house(j:j) = '/'
3331    CONTINUE
3332    CONTINUE
      END IF
      IF((hen_house(i:i) .NE. '/'))hen_house(i+1:i+1) = '/'
      n_files = 0
      DO 3341 i=1,len(egs_home)
        egs_home(i:i) = ' '
3341  CONTINUE
3342  CONTINUE
      call getenv('EGS_HOME',egs_home)
      i = lnblnk1(egs_home)
      IF (( '/' .NE. fool_dec )) THEN
        DO 3351 j=1,i
          IF((egs_home(j:j) .EQ. '/'))egs_home(j:j) = '/'
3351    CONTINUE
3352    CONTINUE
      END IF
      IF((i .GT. 0 .AND. egs_home(i:i) .NE. '/'))egs_home(i+1:i+1) = '/'
      DO 3361 i=1,len(input_file)
        input_file(i:i) = ' '
3361  CONTINUE
3362  CONTINUE
      DO 3371 i=1,len(output_file)
        output_file(i:i) = ' '
3371  CONTINUE
3372  CONTINUE
      DO 3381 i=1,len(work_dir)
        work_dir(i:i) = ' '
3381  CONTINUE
3382  CONTINUE
      DO 3391 i=1,len(pegs_file)
        pegs_file(i:i) = ' '
3391  CONTINUE
3392  CONTINUE
      DO 3401 i=1,len(host_name)
        host_name(i:i) = ' '
3401  CONTINUE
3402  CONTINUE
      n_parallel = 0
      i_parallel = 0
      n_chunk = 0
      is_batch = .false.
      first_parallel = 1
      return
      end
      subroutine egs_combine_runs(combine_routine,extension)
      implicit none
      external combine_routine
      character*(*) extension
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      character*1024 tmp_string,base,command,outfile,parfile_name,base1,
     * text_string
      integer lnblnk1,istat,ipar,egs_system,egs_open_file
      integer*4 i,k,j,numparfiles,textindex
      logical ex,iwin
      iwin=.false.
      DO 3411 i=1,len(base)
        base(i:i) = ' '
3411  CONTINUE
3412  CONTINUE
      base = egs_home(:lnblnk1(egs_home)) // user_code(:lnblnk1(user_cod
     *e)) // '/' // output_file(:lnblnk1(output_file)) // '_w'
      DO 3421 i=1,len(base1)
        base1(i:i) = ' '
3421  CONTINUE
3422  CONTINUE
      base1 = egs_home(:lnblnk1(egs_home)) // user_code(:lnblnk1(user_co
     *de)) // '/' // output_file(:lnblnk1(output_file)) // '_w*' // exte
     *nsion(:lnblnk1(extension))
      DO 3431 i=1,len(outfile)
        outfile(i:i) = ' '
3431  CONTINUE
3432  CONTINUE
      outfile = egs_home(:lnblnk1(egs_home)) // user_code(:lnblnk1(user_
     *code)) // '/' // 'parfiles_tmp'
      DO 3441 i=1,len(command)
        command(i:i) = ' '
3441  CONTINUE
3442  CONTINUE
      command = 'ls ' // base1(:lnblnk1(base1)) // ' | wc -l > ' // outf
     *ile(:lnblnk1(outfile))
      istat = egs_system(command(:lnblnk1(command)))
      IF ((istat.NE.0)) THEN
        command = 'dir ' // base1(:lnblnk1(base1)) // ' | find "File(s)"
     * > ' // outfile(:lnblnk1(outfile))
        istat = egs_system(command(:lnblnk1(command)))
        IF ((istat.NE.0)) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,*) ' Failed to write number of output files from p
     *arallel runs.'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        ELSE
          iwin=.true.
        END IF
      END IF
      ipar=1
      ipar=egs_open_file(ipar,0,1,outfile(:lnblnk1(outfile)))
      IF ((iwin)) THEN
        read(ipar,'(a)',err=3450,end=3450) text_string
        text_string = text_string(:lnblnk1(text_string))
        textindex = index(text_string,'File(s)')
        text_string = text_string(:textindex-1)
        read(text_string,'(i256)',err=3450) numparfiles
      ELSE
        read(ipar,'(i256)',err=3450,end=3450) numparfiles
      END IF
      close(ipar)
      DO 3461 i=1,len(command)
        command(i:i) = ' '
3461  CONTINUE
3462  CONTINUE
      IF ((iwin)) THEN
        command = 'del /Q ' // outfile(:lnblnk1(outfile))
      ELSE
        command = 'rm -f ' // outfile(:lnblnk1(outfile))
      END IF
      istat = egs_system(command(:lnblnk1(command)))
      IF ((istat.NE.0)) THEN
        write(i_log,'(/a)') '***************** Warning: '
        write(i_log,*) ' Failed to delete list of output files from para
     *llel runs.'
      END IF
      k=1
      j=1
3471  IF(j.GT.numparfiles)GO TO 3472
        DO 3481 i=1,len(tmp_string)
          tmp_string(i:i) = ' '
3481    CONTINUE
3482    CONTINUE
        tmp_string = base(:lnblnk1(base))
        call egs_itostring(tmp_string,k,.false.)
        tmp_string = tmp_string(:lnblnk1(tmp_string)) // extension(:lnbl
     *  nk1(extension))
        inquire(file=tmp_string,exist=ex)
        IF (( ex )) THEN
          call combine_routine(tmp_string)
          j=j+1
        END IF
        k=k+1
      GO TO 3471
3472  CONTINUE
      return
3450  write(i_log,'(/a)') '***************** Error: '
      write(i_log,*) ' Failed to read number of output files from parall
     *el runs.'
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
      end
      logical function egs_strip_extension(filen,fext)
      implicit none
      character*(*) filen,fext
      integer l1,l2,lnblnk1,i
      l1 = lnblnk1(filen)
      l2 = lnblnk1(fext)
      IF (( l1 .GE. l2 .AND. filen(l1-l2+1:l1) .EQ. fext(:l2) )) THEN
        egs_strip_extension = .true.
        DO 3491 i=l1-l2+1,len(filen)
          filen(i:i) = ' '
3491    CONTINUE
3492    CONTINUE
      ELSE
        egs_strip_extension = .false.
      END IF
      return
      end
      logical function egs_is_absolute_path(fn)
      implicit none
      character*(*) fn
      integer i,lnblnk1
      DO 3501 i=1,lnblnk1(fn)
        IF (( fn(i:i) .EQ. '/' )) THEN
          egs_is_absolute_path = .true.
          return
        END IF
3501  CONTINUE
3502  CONTINUE
      egs_is_absolute_path = .false.
      return
      end
      integer function egs_get_unit(iunit)
      implicit none
      integer*4 iunit, i
      logical is_open
      IF (( iunit .GT. 0 )) THEN
        inquire(iunit,opened=is_open)
        IF (( .NOT.is_open )) THEN
          egs_get_unit = iunit
          return
        END IF
      END IF
      DO 3511 i=1,99
        inquire(i,opened=is_open)
        IF (( .NOT.is_open )) THEN
          egs_get_unit = i
          return
        END IF
3511  CONTINUE
3512  CONTINUE
      egs_get_unit = -1
      return
      end
      integer function egs_open_file(iunit,rl,action,extension)
      implicit none
      integer*4 iunit, rl, action
      character*(*) extension
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      logical egs_is_absolute_path,is_open
      integer egs_get_unit
      integer i,lnblnk1
      character*1024 tmp_string,error_string
      integer*4 the_unit
      egs_open_file = -1
      the_unit = egs_get_unit(iunit)
      IF (( the_unit .LT. 0 )) THEN
        IF (( action .EQ. 0 )) THEN
          egs_open_file = -1
          return
        END IF
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'No free Fortran I/O units left'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      IF (( egs_is_absolute_path(extension) )) THEN
        inquire(file=extension,opened=is_open)
        IF ((is_open)) THEN
          inquire(file=extension,number=the_unit)
          write(i_log,'(/a)') '***************** Warning: '
          write(i_log,'(a,a,/,a,i3,/,a,/,a)') 'File ',extension(:lnblnk1
     *    (extension)), ' is already opened and connected to unit ',the_
     *    unit, ' Will not try to re-open this file, assuming it has bee
     *n opened', ' by the .io file.'
        ELSE IF(( rl .EQ. 0 )) THEN
          open(the_unit,file=extension,status='unknown')
        ELSE
          open(the_unit,file=extension,status='unknown',form='unformatte
     *d', access='direct', recl=rl)
        END IF
        egs_open_file = the_unit
        return
      END IF
      DO 3521 i=1,len(tmp_string)
        tmp_string(i:i) = ' '
3521  CONTINUE
3522  CONTINUE
      tmp_string = egs_home(:lnblnk1(egs_home)) // user_code(:lnblnk1(us
     *er_code)) // '/' // work_dir(:lnblnk1(work_dir)) // output_file(:l
     *nblnk1(output_file))
      IF (( i_parallel .GT. 0 )) THEN
        tmp_string = tmp_string(:lnblnk1(tmp_string)) // '_w'
        call egs_itostring(tmp_string,i_parallel,.false.)
      END IF
      tmp_string = tmp_string(:lnblnk1(tmp_string)) // extension(:lnblnk
     *1(extension))
      inquire(file=tmp_string,opened=is_open)
      IF ((is_open)) THEN
        inquire(file=tmp_string,number=the_unit)
        write(i_log,'(/a)') '***************** Warning: '
        write(i_log,'(a,a,/,a,i3,/,a,/,a,/)') 'File ',tmp_string(:lnblnk
     *  1(tmp_string)), ' is already opened and connected to unit ',the_
     *  unit, ' Will not try to re-open this file, assuming it has been
     *opened', ' by specifying it in the .io file.'
      ELSE IF(( rl .EQ. 0 )) THEN
        open(the_unit,file=tmp_string,status='unknown',err=3530)
      ELSE
        open(the_unit,file=tmp_string,status='unknown',form='unformatted
     *', access='direct', recl=rl,err=3530)
      END IF
      egs_open_file = the_unit
      return
3530  error_string = 'In egs_open_file: failed to open file ' // tmp_str
     *ing(:lnblnk1(tmp_string)) // char(10) // 'iunit = '
      call egs_itostring(error_string,iunit,.false.)
      error_string = error_string(:lnblnk1(error_string)) // ' the_unit
     *= '
      call egs_itostring(error_string,the_unit,.false.)
      write(i_log,'(/a)') '***************** Error: '
      write(i_log,'(a)') error_string(:lnblnk1(error_string))
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
      end
      integer function egs_open_datfile(iunit,rl,action,extension)
      implicit none
      integer*4 iunit,rl,action
      character*(*) extension
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      integer i,the_unit,lnblnk1,egs_get_unit
      logical egs_is_absolute_path
      character base*1024, fn*1024
      egs_open_datfile = -1
      the_unit = egs_get_unit(iunit)
      IF (( the_unit .LT. 0 )) THEN
        IF (( action .EQ. 0 )) THEN
          egs_open_datfile = -1
          return
        END IF
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'No free Fortran I/O units left'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      IF (( egs_is_absolute_path(extension) )) THEN
        IF (( rl .EQ. 0 )) THEN
          open(the_unit,file=extension,status='old',err=3540)
        ELSE
          open(the_unit,file=extension,status='old',form='unformatted',
     *    access='direct',recl=rl,err=3540)
        END IF
        egs_open_datfile = the_unit
        return
3540    CONTINUE
        IF (( action .EQ. 0 )) THEN
          egs_open_datfile = -2
          return
        END IF
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'Failed to open file ',extension(:lnblnk1(extensi
     *  on))
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      DO 3551 i=1,len(base)
        base(i:i) = ' '
3551  CONTINUE
3552  CONTINUE
      DO 3561 i=1,len(fn)
        fn(i:i) = ' '
3561  CONTINUE
3562  CONTINUE
      base = egs_home(:lnblnk1(egs_home)) // user_code(:lnblnk1(user_cod
     *e)) // '/'
      IF (( i_parallel .GT. 0 )) THEN
        fn = base(:lnblnk1(base)) // output_file(:lnblnk1(output_file))
     *  // '_w'
        call egs_itostring(fn,i_parallel,.false.)
        fn = fn(:lnblnk1(fn)) // extension(:lnblnk1(extension))
      ELSE
        fn = base(:lnblnk1(base)) // output_file(:lnblnk1(output_file))
     *  // extension(:lnblnk1(extension))
      END IF
      IF (( rl .EQ. 0 )) THEN
        open(the_unit,file=fn,status='old',err=3570)
      ELSE
        open(the_unit,file=fn,status='old',form='unformatted',access='di
     *rect', recl=rl,err=3570)
      END IF
      egs_open_datfile = the_unit
      return
3570  CONTINUE
      write(i_log,'(/a)') '***************** Warning: '
      write(i_log,'(a,a)') 'Failed to open ',fn(:lnblnk1(fn))
      DO 3581 i=1,len(fn)
        fn(i:i) = ' '
3581  CONTINUE
3582  CONTINUE
      IF (( i_parallel .GT. 0 )) THEN
        fn = base(:lnblnk1(base)) // input_file(:lnblnk1(input_file)) //
     *   '_w'
        call egs_itostring(fn,i_parallel,.false.)
        fn = fn(:lnblnk1(fn)) // extension(:lnblnk1(extension))
      ELSE
        fn = base(:lnblnk1(base)) // input_file(:lnblnk1(input_file)) //
     *   extension(:lnblnk1(extension))
      END IF
      IF (( rl .EQ. 0 )) THEN
        open(the_unit,file=fn,status='old',err=3590)
      ELSE
        open(the_unit,file=fn,status='old',form='unformatted',access='di
     *rect', recl=rl,err=3590)
      END IF
      egs_open_datfile = the_unit
      return
3590  CONTINUE
      write(i_log,'(/a)') '***************** Error: '
      write(i_log,*) 'Failed to open data file'
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
      end
      integer function egs_open_file_junk(iunit,do_it_anyway,filen)
      implicit none
      integer*4 iunit
      logical do_it_anyway
      character*(*) filen
      logical aux
      integer*4 the_unit,i
      inquire(file=filen,exist=aux)
      IF (( .NOT.aux )) THEN
        egs_open_file_junk = -2
        return
      END IF
      IF (( iunit .LT. 0 )) THEN
        the_unit = -iunit
      ELSE
        the_unit = iunit
      END IF
      IF (( the_unit .NE. 0 )) THEN
        inquire(unit=the_unit,opened=aux)
        IF (( aux )) THEN
          IF (( .NOT.do_it_anyway )) THEN
            egs_open_file_junk = -4
            return
          END IF
          IF((iunit .LT. 0))the_unit = 0
        END IF
      END IF
      IF (( the_unit .EQ. 0 )) THEN
        DO 3601 i=1,99
          inquire(unit=i,opened=aux)
          IF (( .NOT.aux )) THEN
            the_unit = i
            GO TO3602
          END IF
3601    CONTINUE
3602    CONTINUE
        IF (( the_unit .EQ. 0 )) THEN
          egs_open_file_junk = -1
          return
        END IF
      END IF
      open(the_unit,file=filen,status='old',err=3610)
      egs_open_file_junk = the_unit
      return
3610  egs_open_file_junk = -3
      return
      end
      subroutine egs_strip_path(fname)
      implicit none
      character*(*) fname
      integer i,l,l1,lnblnk1,j
      character slash
      slash = '/'
      l = lnblnk1(fname)
      DO 3621 i=1,l
        IF (( fname(i:i) .EQ. slash )) THEN
          fname(i:i) = '/'
        END IF
3621  CONTINUE
3622  CONTINUE
      DO 3631 i=l,1,-1
        IF (( fname(i:i) .EQ. '/' .OR. fname(i:i) .EQ. slash )) THEN
          l1 = l-i
          fname(:l1) = fname(i+1:l)
          DO 3641 j=l1+1,len(fname)
            fname(j:j) = ' '
3641      CONTINUE
3642      CONTINUE
          return
        END IF
3631  CONTINUE
3632  CONTINUE
      return
      end
      subroutine replace_env(fname)
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      character*(*) fname
      character*256 dirname
      integer indsep,ind1,ind2
      indsep = index(fname,'/')
      IF((indsep .LE. 0))return
      ind1=index(fname,'$')
      ind2=index(fname,'~')
      IF ((ind1.EQ.1)) THEN
        call getenv(fname(2:indsep-1),dirname)
        IF ((dirname.EQ.' ')) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,'(a,a/,a)') ' Error in file name: ',fname(:lnblnk1
     *    (fname)), ' First element in name does not specify a defined e
     *nvironment variable.'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        fname=dirname(:lnblnk1(dirname))//fname(indsep:)
        write(i_log,'(//a,a/)') ' Retrieving file: ',fname(:lnblnk1(fnam
     *  e))
      ELSE IF((ind2.EQ.1)) THEN
        call getenv('HOME',dirname)
        IF ((dirname.EQ.' ')) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,'(a,a/,a)') ' Error in file name: ',fname(:lnblnk1
     *    (fname)), ' HOME is undefined.'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        fname=dirname(:lnblnk1(dirname))//fname(indsep:)
        write(i_log,'(//a,a/)') ' Retrieving file: ',fname(:lnblnk1(fnam
     *  e))
      END IF
      return
      end
      subroutine egs_get_usercode(ucode)
      implicit none
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      character*(*) ucode
      character*512 arg
      integer l,l1,lnblnk1,i
      call getarg(0,arg)
      call egs_strip_path(arg)
      l = lnblnk1(arg)
      IF (( arg(l-3:l) .EQ. '.exe' )) THEN
        arg(l-3:l) = ' '
        l = l - 4
      END IF
      IF (( arg(l-5:l) .EQ. '_debug' )) THEN
        arg(l-5:l) = ' '
        l = l-5
      END IF
      IF (( arg(l-5:l) .EQ. '_noopt' )) THEN
        arg(l-5:l) = ' '
        l = l-5
      END IF
      l1 = len(ucode)
      IF (( l .GT. l1 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) ' user code name is too long (',l,' chars)'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      DO 3651 i=1,len(ucode)
        ucode(i:i) = ' '
3651  CONTINUE
3652  CONTINUE
      ucode(:l) = arg(:l)
      return
      end
      subroutine egs_itostring(string,i,leave_space)
      implicit none
      character*(*) string
      integer*4 i
      integer l,lnblnk1,idiv,itmp,iaux
      logical first,leave_space
      l = lnblnk1(string)+1
      IF((l .GT. 1 .AND. leave_space))l=l+1
      idiv = 1000000000
      itmp = i
      first = .false.
      do while(idiv.gt.0)
      iaux = itmp/idiv
      IF (( (iaux .GT. 0 .OR. first ) .AND. l .LE. len(string) )) THEN
        string(l:l) = char(iaux+48)
        first = .true.
        l = l+1
      END IF
      itmp = itmp - iaux*idiv
      idiv = idiv/10
      end do
      return
      end
      real*8 function egs_rndm()
      implicit none
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      IF (( rng_seed .GT. 24 )) THEN
        call ranlux(rng_array)
        rng_seed = 1
      END IF
      egs_rndm = rng_array(rng_seed)
      rng_seed = rng_seed + 1
      return
      end
      integer function egs_add_medium(medname)
      implicit none
      character*(*) medname
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      integer*4 i,l,imed,medname_len
      character c
      logical same
      l = min(len(medname),24)
      medname_len = l
      DO 3661 i=1,l
        c = medname(i:i)
        IF (( ichar(c) .EQ. 0 )) THEN
          medname_len = i-1
          GO TO3662
        END IF
3661  CONTINUE
3662  CONTINUE
      DO 3671 imed=1,nmed
        l = 24
        DO 3681 i=1,24
          IF (( media(i,imed)(1:1) .EQ. ' ' )) THEN
            l = i-1
            GO TO3682
          END IF
3681    CONTINUE
3682    CONTINUE
        IF (( l .EQ. medname_len )) THEN
          same = .true.
          DO 3691 i=1,l
            c = medname(i:i)
            IF (( c .NE. media(i,imed)(1:1) )) THEN
              same = .false.
              GO TO3692
            END IF
3691      CONTINUE
3692      CONTINUE
          IF (( same )) THEN
            egs_add_medium = imed
            return
          END IF
        END IF
3671  CONTINUE
3672  CONTINUE
      nmed = nmed + 1
      IF (( nmed .GT. 1 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,'(a,/,a,i3,a)') 'In egs_add_medium: maximum number o
     *f media exceeded ', 'Increase the macro $MXMED (currently ',1,') a
     *nd retry'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      l = min(len(medname),24)
      DO 3701 i=1,l
        c = medname(i:i)
        IF (( ichar(c) .EQ. 0 )) THEN
          l = i-1
          GO TO3702
        END IF
        media(i,nmed) = ' '
        media(i,nmed)(1:1) = c
3701  CONTINUE
3702  CONTINUE
      IF (( l .LT. 24 )) THEN
        DO 3711 i=l+1,24
          media(i,nmed) = ' '
3711    CONTINUE
3712    CONTINUE
      END IF
      egs_add_medium = nmed
      return
      end
      subroutine egs_get_medium_name(imed,medname)
      implicit none
      character*(*) medname
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      integer*4 i,l,imed
      DO 3721 i=1,len(medname)
        medname(i:i) = ' '
3721  CONTINUE
3722  CONTINUE
      IF (( imed .LT. 1 .OR. imed .GT. nmed )) THEN
        return
      END IF
      l = 24
      DO 3731 l=24,1,-1
        IF((media(l,imed)(1:1) .NE. ' '))GO TO3732
3731  CONTINUE
3732  CONTINUE
      l = min(l,len(medname))
      DO 3741 i=1,l
        medname(i:i) = media(i,imed)(1:1)
3741  CONTINUE
3742  CONTINUE
      return
      end
      subroutine egs_get_electron_data(func,imed,which)
      implicit none
      integer*4 imed,which
      external func
      COMMON/ELECIN/ esig_e(1),psig_e(1), esige_max, psige_max, range_ep
     *(0:1,500,1), E_array(500,1), etae_ms0(500,1),etae_ms1(500,1),etap_
     *ms0(500,1),etap_ms1(500,1),q1ce_ms0(500,1),q1ce_ms1(500,1),q1cp_ms
     *0(500,1),q1cp_ms1(500,1),q2ce_ms0(500,1),q2ce_ms1(500,1),q2cp_ms0(
     *500,1),q2cp_ms1(500,1),blcce0(500,1),blcce1(500,1), EKE0(1),EKE1(1
     *), XR0(1),TEFF0(1),BLCC(1),XCC(1), ESIG0(500,1),ESIG1(500,1),PSIG0
     *(500,1),PSIG1(500,1),EDEDX0(500,1),EDEDX1(500,1),PDEDX0(500,1),PDE
     *DX1(500,1),EBR10(500,1),EBR11(500,1),PBR10(500,1),PBR11(500,1),PBR
     *20(500,1),PBR21(500,1),TMXS0(500,1),TMXS1(500,1), expeke1(1), IUNR
     *ST(1),EPSTFL(1),IAPRIM(1), sig_ismonotone(0:1,1)
      real*8 esig_e,   psig_e,   esige_max,  psige_max,  range_ep,  E_ar
     *ray,  etae_ms0,etae_ms1,  etap_ms0,etap_ms1,  q1ce_ms0,q1ce_ms1,
     *q1cp_ms0,q1cp_ms1,  q2ce_ms0,q2ce_ms1,  q2cp_ms0,q2cp_ms1,  blcce0
     *,blcce1,   expeke1,  EKE0,EKE1, XR0,  TEFF0,  BLCC,  XCC,  ESIG0,E
     *SIG1,  PSIG0,PSIG1,  EDEDX0,EDEDX1,  PDEDX0,PDEDX1,  EBR10,EBR11,
     * PBR10,PBR11,  PBR20,PBR21,  TMXS0,TMXS1
      integer*4 IUNRST,  EPSTFL,  IAPRIM
      logical sig_ismonotone
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      real*8 lemin,lemax
      lemin = (1 - eke0(imed))/eke1(imed)
      lemax = (meke(imed) - eke0(imed))/eke1(imed)
      IF (( which .EQ. 1 )) THEN
        call func(meke(imed),lemin,lemax,esig0(1,imed),esig1(1,imed))
      ELSE IF(( which .EQ. 2 )) THEN
        call func(meke(imed),lemin,lemax,psig0(1,imed),psig1(1,imed))
      ELSE IF(( which .EQ. 3 )) THEN
        call func(meke(imed),lemin,lemax,ededx0(1,imed),ededx1(1,imed))
      ELSE IF(( which .EQ. 4 )) THEN
        call func(meke(imed),lemin,lemax,pdedx0(1,imed),pdedx1(1,imed))
      ELSE IF(( which .EQ. 5 )) THEN
        call func(meke(imed),lemin,lemax,ebr10(1,imed),ebr11(1,imed))
      ELSE IF(( which .EQ. 6 )) THEN
        call func(meke(imed),lemin,lemax,pbr10(1,imed),pbr11(1,imed))
      ELSE IF(( which .EQ. 7 )) THEN
        call func(meke(imed),lemin,lemax,pbr20(1,imed),pbr21(1,imed))
      ELSE IF(( which .EQ. 8 )) THEN
        call func(meke(imed),lemin,lemax,tmxs0(1,imed),tmxs1(1,imed))
      ELSE IF(( which .EQ. 9 )) THEN
        call func(meke(imed),lemin,lemax,range_ep(0,1,imed),range_ep(1,1
     *  ,imed))
      ELSE
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'Unknown electron data type ',which
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      return
      end
      subroutine egs_get_photon_data(func,imed,which)
      implicit none
      integer*4 imed,which
      external func
      COMMON/PHOTIN/ EBINDA(1), GE0(1),GE1(1), GMFP0(2000,1),GMFP1(2000,
     *1),GBR10(2000,1),GBR11(2000,1),GBR20(2000,1),GBR21(2000,1), RCO0(1
     *),RCO1(1), RSCT0(100,1),RSCT1(100,1), COHE0(2000,1),COHE1(2000,1),
     *  PHOTONUC0(2000,1),PHOTONUC1(2000,1), DPMFP, MPGEM(1,1), NGR(1)
      real*8 EBINDA,  GE0,GE1,  GMFP0,GMFP1,  GBR10,GBR11,  GBR20,GBR21,
     *  RCO0,RCO1,  RSCT0,RSCT1,  COHE0,COHE1,   PHOTONUC0,PHOTONUC1,  D
     *PMFP
      integer*4
     *                  MPGEM,
     *                          NGR
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      real*8 lemin,lemax
      lemin = (1 - ge0(imed))/ge1(imed)
      lemax = (mge(imed) - ge0(imed))/ge1(imed)
      IF (( which .EQ. 1 )) THEN
        call func(mge(imed),lemin,lemax,gmfp0(1,imed),gmfp1(1,imed))
      ELSE IF(( which .EQ. 2 )) THEN
        call func(mge(imed),lemin,lemax,gbr10(1,imed),gbr11(1,imed))
      ELSE IF(( which .EQ. 3 )) THEN
        call func(mge(imed),lemin,lemax,gbr20(1,imed),gbr21(1,imed))
      ELSE IF(( which .EQ. 4 )) THEN
        call func(mge(imed),lemin,lemax,cohe0(1,imed),cohe1(1,imed))
      ELSE IF(( which .EQ. 5 )) THEN
        call func(mge(imed),lemin,lemax,PHOTONUC0(1,imed),PHOTONUC1(1,im
     *  ed))
      ELSE
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'Unknown photon data type ',which
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      return
      end
      subroutine egs_print_binding_energies
      implicit none
      COMMON/EDGE/binding_energies(30,100), interaction_prob(6,100), rel
     *axation_prob(39,100), edge_energies(16,100), edge_number(100), edg
     *e_a(16,100), edge_b(16,100), edge_c(16,100), edge_d(16,100), IEDGF
     *L(1),IPHTER(1)
      real*8 binding_energies,  interaction_prob,    relaxation_prob,  e
     *dge_energies,  edge_a,edge_b,edge_c,edge_d
      integer*2 IEDGFL,  IPHTER
      integer*4 edge_number
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      integer*4 i,j
      integer*4 lnblnk1
      character*3 labels(16)
      data labels/'  K',' L1',' L2',' L3', ' M1',' M2',' M3',' M4',' M5'
     *, ' N1',' N2',' N3',' N4',' N5',' N6',' N7'/
      write(i_log,'(a,a,a)') 'Binding energies from ',photon_xsections(:
     *lnblnk1(photon_xsections)), ' photon cross section library'
      DO 3751 j=1,100
        DO 3761 i=1,16
          IF (( binding_energies(i,j) .GT. 0 )) THEN
            write(i_log,'(a,i3,a,a,a,1pe12.4,a)') ' Eb(',j,',',labels(i)
     *      ,') = ',binding_energies(i,j),' MeV'
          END IF
3761    CONTINUE
3762    CONTINUE
3751  CONTINUE
3752  CONTINUE
      return
      end
      subroutine egs_scale_xcc(imed,factor)
      implicit none
      integer*4 imed
      real*8 factor
      COMMON/ELECIN/ esig_e(1),psig_e(1), esige_max, psige_max, range_ep
     *(0:1,500,1), E_array(500,1), etae_ms0(500,1),etae_ms1(500,1),etap_
     *ms0(500,1),etap_ms1(500,1),q1ce_ms0(500,1),q1ce_ms1(500,1),q1cp_ms
     *0(500,1),q1cp_ms1(500,1),q2ce_ms0(500,1),q2ce_ms1(500,1),q2cp_ms0(
     *500,1),q2cp_ms1(500,1),blcce0(500,1),blcce1(500,1), EKE0(1),EKE1(1
     *), XR0(1),TEFF0(1),BLCC(1),XCC(1), ESIG0(500,1),ESIG1(500,1),PSIG0
     *(500,1),PSIG1(500,1),EDEDX0(500,1),EDEDX1(500,1),PDEDX0(500,1),PDE
     *DX1(500,1),EBR10(500,1),EBR11(500,1),PBR10(500,1),PBR11(500,1),PBR
     *20(500,1),PBR21(500,1),TMXS0(500,1),TMXS1(500,1), expeke1(1), IUNR
     *ST(1),EPSTFL(1),IAPRIM(1), sig_ismonotone(0:1,1)
      real*8 esig_e,   psig_e,   esige_max,  psige_max,  range_ep,  E_ar
     *ray,  etae_ms0,etae_ms1,  etap_ms0,etap_ms1,  q1ce_ms0,q1ce_ms1,
     *q1cp_ms0,q1cp_ms1,  q2ce_ms0,q2ce_ms1,  q2cp_ms0,q2cp_ms1,  blcce0
     *,blcce1,   expeke1,  EKE0,EKE1, XR0,  TEFF0,  BLCC,  XCC,  ESIG0,E
     *SIG1,  PSIG0,PSIG1,  EDEDX0,EDEDX1,  PDEDX0,PDEDX1,  EBR10,EBR11,
     * PBR10,PBR11,  PBR20,PBR21,  TMXS0,TMXS1
      integer*4 IUNRST,  EPSTFL,  IAPRIM
      logical sig_ismonotone
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      IF (( imed .GT. 0 .AND. imed .LE. nmed )) THEN
        xcc(imed) = xcc(imed)*factor
      END IF
      return
      end
      subroutine egs_write_string(ounit,string)
      implicit none
      integer*4 ounit
      character*(*) string
      write(ounit,'(a,$)') string
      call flush(ounit)
      return
      end
      subroutine egs_swap_2(c)
      character c(2),tmp
      tmp=c(2)
      c(2)=c(1)
      c(1)=tmp
      return
      end
      subroutine egs_swap_4(c)
      character c(4),tmp
      tmp=c(4)
      c(4)=c(1)
      c(1)=tmp
      tmp=c(3)
      c(3)=c(2)
      c(2)=tmp
      return
      end
      subroutine set_spline(x,f,a,b,c,d,n)
      implicit none
      integer*4 n
      real*8 x(n),f(n),a(n),b(n),c(n),d(n)
      integer*4 m1,m2,m,mr
      real*8 s,r
      m1 = 2
      m2 = n-1
      s = 0
      DO 3771 m=1,m2
        d(m) = x(m+1) - x(m)
        r = (f(m+1) - f(m))/d(m)
        c(m) = r - s
        s = r
3771  CONTINUE
3772  CONTINUE
      s=0
      r=0
      c(1)=0
      c(n)=0
      DO 3781 m=m1,m2
        c(m) = c(m) + r*c(m-1)
        b(m) = 2*(x(m-1) - x(m+1)) - r*s
        s = d(m)
        r = s/b(m)
3781  CONTINUE
3782  CONTINUE
      mr = m2
      DO 3791 m=m1,m2
        c(mr) = (d(mr)*c(mr+1) - c(mr))/b(mr)
        mr = mr - 1
3791  CONTINUE
3792  CONTINUE
      DO 3801 m=1,m2
        s = d(m)
        r = c(m+1) - c(m)
        d(m) = r/s
        c(m) = 3*c(m)
        b(m) = (f(m+1)-f(m))/s - (c(m)+r)*s
        a(m) = f(m)
3801  CONTINUE
3802  CONTINUE
      return
      end
      real*8 function spline(s,x,a,b,c,d,n)
      implicit none
      integer*4 n
      real*8 s,x(n),a(n),b(n),c(n),d(n)
      integer m_lower,m_upper,direction,m,ml,mu,mav
      real*8 q
      IF (( x(1) .GT. x(n) )) THEN
        direction = 1
        m_lower = n
        m_upper = 0
      ELSE
        direction = 0
        m_lower = 0
        m_upper = n
      END IF
      IF (( s .GE. x(m_upper + direction) )) THEN
        m = m_upper + 2*direction - 1
      ELSE IF(( s .LE. x(m_lower+1-direction) )) THEN
        m = m_lower - 2*direction + 1
      ELSE
        ml = m_lower
        mu = m_upper
3811    IF(iabs(mu-ml).LE.1)GO TO 3812
          mav = (ml+mu)/2
          IF (( s .LT. x(mav) )) THEN
            mu = mav
          ELSE
            ml = mav
          END IF
        GO TO 3811
3812    CONTINUE
        m = mu + direction - 1
      END IF
      q = s - x(m)
      spline = a(m) + q*(b(m) + q*(c(m) + q*d(m)))
      return
      end
      subroutine prepare_alias_table(nsbin,xs_array,fs_array,ws_array,ib
     *in_array)
      implicit none
      integer nsbin
      integer*4 ibin_array(nsbin)
      real*8 xs_array(0:nsbin),fs_array(0:nsbin),ws_array(nsbin)
      integer*4 i,j_l,j_h
      real*8 sum,aux
      sum = 0
      DO 3821 i=1,nsbin
        aux = 0.5*(fs_array(i)+fs_array(i-1))*(xs_array(i)-xs_array(i-1)
     *  )
        IF((aux .LT. 1e-30))aux = 1e-30
        ws_array(i) = -aux
        ibin_array(i) = 1
        sum = sum + aux
3821  CONTINUE
3822  CONTINUE
      sum = sum/nsbin
      DO 3831 i=1,nsbin-1
        DO 3841 j_h=1,nsbin
          IF (( ws_array(j_h) .LT. 0 )) THEN
            IF((abs(ws_array(j_h)) .GT. sum))GOTO 3850
          END IF
3841    CONTINUE
3842    CONTINUE
        j_h = nsbin
3850    CONTINUE
          DO 3851 j_l=1,nsbin
          IF (( ws_array(j_l) .LT. 0 )) THEN
            IF((abs(ws_array(j_l)) .LT. sum))GOTO 3860
          END IF
3851    CONTINUE
3852    CONTINUE
        j_l = nsbin
3860    aux = sum - abs(ws_array(j_l))
        ws_array(j_h) = ws_array(j_h) + aux
        ws_array(j_l) = -ws_array(j_l)/sum
        ibin_array(j_l) = j_h
        IF((i .EQ. nsbin-1))ws_array(j_h) = 1
3831  CONTINUE
3832  CONTINUE
      return
      end
      real*8 function alias_sample1(nsbin,xs_array,fs_array,ws_array,ibi
     *n_array)
      implicit none
      integer nsbin
      integer*4 ibin_array(nsbin)
      real*8 xs_array(0:nsbin),fs_array(0:nsbin),ws_array(nsbin)
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      integer*4 j
      real*8 r1,r2,aj,x,dx,a,rnno1
      IF (( rng_seed .GT. 24 )) THEN
        call ranlux(rng_array)
        rng_seed = 1
      END IF
      r1 = rng_array(rng_seed)
      rng_seed = rng_seed + 1
      IF (( rng_seed .GT. 24 )) THEN
        call ranlux(rng_array)
        rng_seed = 1
      END IF
      r2 = rng_array(rng_seed)
      rng_seed = rng_seed + 1
      aj = 1 + r1*nsbin
      j = aj
      aj = aj - j
      IF((aj .GT. ws_array(j)))j = ibin_array(j)
      x = xs_array(j-1)
      dx = xs_array(j)-x
      IF (( fs_array(j-1) .GT. 0 )) THEN
        a = fs_array(j)/fs_array(j-1)-1
        IF (( abs(a) .LT. 0.2 )) THEN
          rnno1 = 0.5*(1-r2)*a
          alias_sample1 = x + r2*dx*(1+rnno1*(1-r2*a))
        ELSE
          alias_sample1 = x - dx/a*(1-sqrt(1+r2*a*(2+a)))
        END IF
      ELSE
        alias_sample1 = x + dx*sqrt(r2)
      END IF
      return
      end
      subroutine prepare_alias_histogram(nsbin,ws_array,ibin_array)
      implicit none
      integer*4 nsbin,ibin_array(nsbin)
      real*8 ws_array(nsbin)
      integer*4 i,j_l,j_h
      real*8 sum,aux
      sum = 0
      DO 3871 i=1,nsbin
        sum = sum + ws_array(i)
        ibin_array(i) = -1
3871  CONTINUE
3872  CONTINUE
      sum = sum/nsbin
      DO 3881 i=1,nsbin-1
        DO 3891 j_h=1,nsbin
          IF((ibin_array(j_h) .LT. 0 .AND. ws_array(j_h) .GT. sum))GO TO
     *    3892
3891    CONTINUE
3892    CONTINUE
        DO 3901 j_l=1,nsbin
          IF((ibin_array(j_l) .LT. 0 .AND. ws_array(j_l) .LT. sum))GO TO
     *    3902
3901    CONTINUE
3902    CONTINUE
        aux = sum - ws_array(j_l)
        ws_array(j_h) = ws_array(j_h) - aux
        ws_array(j_l) = ws_array(j_l)/sum
        ibin_array(j_l) = j_h
3881  CONTINUE
3882  CONTINUE
      DO 3911 i=1,nsbin
        IF (( ibin_array(i) .LT. 0 )) THEN
          ibin_array(i) = i
          ws_array(i) = 1
        END IF
3911  CONTINUE
3912  CONTINUE
      return
      end
      integer*4 function sample_alias_histogram(nsbin,ws_array,ibin_arra
     *y)
      implicit none
      integer*4 nsbin,ibin_array(*)
      real*8 ws_array(*)
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      real*8 r1,r2
      integer*4 ibin
      IF (( rng_seed .GT. 24 )) THEN
        call ranlux(rng_array)
        rng_seed = 1
      END IF
      r1 = rng_array(rng_seed)
      rng_seed = rng_seed + 1
      IF (( rng_seed .GT. 24 )) THEN
        call ranlux(rng_array)
        rng_seed = 1
      END IF
      r2 = rng_array(rng_seed)
      rng_seed = rng_seed + 1
      ibin = 1 + nsbin*r1
      IF((r2 .GT. ws_array(ibin)))ibin = ibin_array(ibin)
      sample_alias_histogram = ibin
      return
      end
      subroutine gauss_legendre(x1,x2,x,w,n)
      implicit none
      integer*4 n
      real*8 x1,x2,x(n),w(n)
      real*8 eps,Pi
      parameter (eps = 3.D-14,Pi=3.141592654D0)
      integer*4 i,m,j
      real*8 xm,xl,z,z1,p1,p2,p3,pp
      m = (n + 1)/2
      xm=0.5d0*(x2+x1)
      xl=0.5d0*(x2-x1)
      DO 3921 i=1,m
        z=cos(Pi*(i-.25d0)/(n+.5d0))
3931    CONTINUE
          p1=1.d0
          p2=0.d0
          DO 3941 j=1,n
            p3 = p2
            p2 = p1
            p1=((2.d0*j-1.d0)*z*p2-(j-1.d0)*p3)/j
3941      CONTINUE
3942      CONTINUE
          pp=n*(z*p1-p2)/(z*z-1.d0)
          z1=z
          z=z1-p1/pp
          IF(((abs(z-z1) .LT. eps)))GO TO3932
        GO TO 3931
3932    CONTINUE
        x(i)=xm-xl*z
        x(n+1-i)=xm+xl*z
        w(i)=2.d0*xl/((1.d0-z*z)*pp*pp)
        w(n+1-i)=w(i)
3921  CONTINUE
3922  CONTINUE
      return
      end
      integer function lnblnk1(string)
      character*(*) string
      integer i
      DO 3951 i=len(string),1,-1
        j = ichar(string(i:i))
        IF (( j .EQ. 0 )) THEN
          lnblnk1 = i-1
          return
        END IF
        IF (( j .NE. 9 .AND. j .NE. 10 .AND. j .NE. 11 .AND. j .NE. 12 .
     *  AND. j .NE. 13 .AND. j .NE. 32 )) THEN
          lnblnk1 = i
          return
        END IF
3951  CONTINUE
3952  CONTINUE
      lnblnk1 = 0
      return
      end
      real*8 FUNCTION ERF1(X)
      implicit none
      real*8 x
      double precision A(0:22,2)
      double precision CONST,  BN,BN1,BN2,  Y,FAC
      integer*4 N,  K,  NLIM(2)
      DATA A/ 1.0954712997776232 , -0.2891754011269890 , 0.1104563986337
     *951 , -0.0412531882278565 , 0.0140828380706516 , -0.00432929544743
     *14 , 0.0011982719015923 , -0.0002999729623532 , 0.0000683258603789
     * , -0.0000142469884549 , 0.0000027354087728 , -0.0000004861912872
     *, 0.0000000803872762 , -0.0000000124184183 , 0.0000000017995326 ,
     *-0.0000000002454795 , 0.0000000000316251 , -0.0000000000038590 , 0
     *.0000000000004472 , -0.0000000000000493 , 0.0000000000000052 , -0.
     *0000000000000005 , 0.0000000000000001 , 0.9750834237085559 , -0.02
     *40493938504146 , 0.0008204522408804 , -0.0000434293081303 , 0.0000
     *030184470340 , -0.0000002544733193 , 0.0000000248583530 , -0.00000
     *00027317201 , 0.0000000003308472 , 0.0000000000001464 , -0.0000000
     *000000244 , 0.0000000000000042 , -0.0000000000000008 , 0.000000000
     *0000001 , 9*0.0 /
      DATA NLIM/ 22,16 /
      DATA CONST/ 1.128379167095513 /
      IF (( x .GT. 3 )) THEN
        y = 3/x
        k = 2
      ELSE
        y = x/3
        k = 1
      END IF
      FAC = 2.0 * ( 2.0 * Y*Y - 1.0 )
      BN1 = 0.0
      BN = 0.0
      DO 3961 n=NLIM(K),0,-1
        BN2 = BN1
        BN1 = BN
        BN = FAC * BN1 - BN2 + A(N,K)
3961  CONTINUE
3962  CONTINUE
      IF (( k .EQ. 1 )) THEN
        erf1 = CONST * Y * ( BN - BN1 )
      ELSE
        erf1 = 1 - CONST * EXP(-X**2) * ( BN - BN2 + A(0,K) )/(4.0 * X)
      END IF
      RETURN
      end
      real*8 FUNCTION ZERO()
      implicit none
      integer*4 i
      real*8 x, xtemp
      x = 1.E-20
      DO 3971 i=1,100
        IF ((x .EQ. 0.0)) THEN
          GO TO3972
        ELSE
          xtemp = x
        END IF
        x = x/1.E5
3971  CONTINUE
3972  CONTINUE
      x = xtemp
      DO 3981 i=1,5
        IF ((x .NE. 0.0)) THEN
          xtemp = x
        ELSE
          GO TO3982
        END IF
        x = x/10
3981  CONTINUE
3982  CONTINUE
      x = xtemp
      DO 3991 i=2,10
        IF ((x .NE. 0.0)) THEN
          xtemp = x
        ELSE
          GO TO3992
        END IF
        x = x/i
3991  CONTINUE
3992  CONTINUE
      zero = xtemp
      return
      end
      character*512 function toUpper(a_string)
      character*(*) a_string
      character*512 the_string
      integer*4 cursor, i, lnblnk1
      toUpper = a_string
      the_string = a_string
      DO 4001 i=1,lnblnk1(the_string)
        cursor=ICHAR(the_string(i:i))
        IF (((cursor.GE.97).AND.(cursor.LE.122))) THEN
          cursor=cursor-32
          toUpper(i:i)=CHAR(cursor)
        END IF
4001  CONTINUE
4002  CONTINUE
      return
      end
      integer*1 function egs_read_byte(iunit, jrec)
      implicit none
      integer iunit, jrec, i, j, ierr
      integer*1 i_1
      character c_1
      equivalence (i_1,c_1)
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      read(iunit,rec=jrec,IOSTAT=ierr) c_1
      IF ((ierr.ne.0)) THEN
        write(i_log,'(/a)') '***************** Warning: '
        write(i_log,*) ' *** egs_read_byte: ERROR READING A byte *** '
        write(i_log,*) ' From unit ',iunit,' position ',jrec,' bytes'
        egs_read_byte = -1
        return
      END IF
      jrec = jrec + 1
      egs_read_byte = i_1
      return
      end
      integer*2 function egs_read_short(iunit, jrec)
      implicit none
      integer iunit, jrec, i, j, ierr
      integer*2 i_2
      character c_2(2)
      equivalence (i_2,c_2)
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      j = 0
      DO 4011 i=jrec,jrec+1
        j = j + 1
        read(iunit,rec=i,IOSTAT=ierr) c_2(j)
        IF ((ierr.ne.0)) THEN
          write(i_log,'(/a)') '***************** Warning: '
          write(i_log,*) ' *** egs_read_short: ERROR READING short integ
     *er *** '
          write(i_log,*) ' From unit ',iunit,' position ',jrec,' bytes'
          egs_read_short = -1
          return
        END IF
4011  CONTINUE
4012  CONTINUE
      jrec = jrec + 2
      egs_read_short = i_2
      return
      end
      integer*4 function egs_read_int(iunit, jrec)
      implicit none
      integer iunit, jrec, i, j, ierr
      integer*4 i_4
      character c_4(4)
      equivalence (i_4,c_4)
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      j = 0
      DO 4021 i=jrec,jrec+3
        j = j + 1
        read(iunit,rec=i,IOSTAT=ierr) c_4(j)
        IF ((ierr.ne.0)) THEN
          write(i_log,'(/a)') '***************** Warning: '
          write(i_log,*) ' *** egs_read_int: ERROR READING integer *** '
          write(i_log,*) ' From unit ',iunit,' position ',jrec,' bytes'
          egs_read_int = -1
          return
        END IF
4021  CONTINUE
4022  CONTINUE
      jrec = jrec + 4
      egs_read_int = i_4
      return
      end
      real*4 function egs_read_real(iunit, jrec)
      implicit none
      integer iunit, jrec, i, j, ierr
      real*4 r_4
      character c_4(4)
      equivalence (r_4,c_4)
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      j = 0
      DO 4031 i=jrec,jrec+3
        j = j + 1
        read(iunit,rec=i,IOSTAT=ierr) c_4(j)
        IF ((ierr.ne.0)) THEN
          write(i_log,'(/a)') '***************** Warning: '
          write(i_log,*) ' *** egs_read_real: ERROR READING float *** '
          write(i_log,*) ' From unit ',iunit,' position ',jrec,' bytes'
          egs_read_real = -1
          return
        END IF
4031  CONTINUE
4032  CONTINUE
      jrec = jrec + 4
      egs_read_real = r_4
      return
      end
      integer*4 function ibsearch(a, nsh, b)
      implicit none
      real*8 a, b(*)
      integer*4 min,max,help,nsh
      real*8 x
      min = 1
      max = nsh
      x = a
4041  IF(min.GE.max-1)GO TO 4042
        help = (max+min)/2
        IF (( b(help).le.x)) THEN
          min = help
        ELSE
          max = help
        END IF
      GO TO 4041
4042  CONTINUE
      ibsearch = min
      return
      end
      SUBROUTINE ANNIH
      implicit none
      COMMON/QDEBUG/QDEBUG
      LOGICAL QDEBUG
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/UPHIOT/THETA,SINTHE,COSTHE,SINPHI, COSPHI,PI,TWOPI,PI5D2
      real*8 THETA,  SINTHE,  COSTHE,  SINPHI,  COSPHI,  PI,TWOPI,PI5D2
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      common/egs_vr/ e_max_rr(1),  prob_RR,  nbr_split,  i_play_RR,
     * i_survived_RR,
     *     n_RR_warning,                                        i_do_rr(
     *1)
      real*8          e_max_rr,prob_RR
      integer*4       nbr_split,i_play_RR,i_survived_RR,n_RR_warning
      integer*2     i_do_rr
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      DOUBLE PRECISION PAVIP,  PESG1,  PESG2
      real*8 AVIP,  A,                  G,T,P,                      POT,
     *
     *     EP0,                                                 WSAMP,
     *                       RNNO01,
     *                     RNNO02,
     *                                   EP,
     * REJF,                                                       ESG1,
     *                                      ESG2,
     *               aa,bb,cc,sinpsi,sindel,cosdel,us,vs,cphi,sphi
      integer*4
     *                     ibr
      real*8 xphi,xphi2,yphi,yphi2,rhophi2
      integer*4 ip
      NPold = NP
      IF (( nbr_split .LE. 0 )) THEN
        return
      END IF
      PAVIP=E(NP)+PRM
      AVIP=PAVIP
      A=AVIP/RM
      G=A-1.0
      T=G-1.0
      P=SQRT(A*T)
      POT=P/T
      EP0=1.0/(A+P)
      WSAMP=LOG((1.0-EP0)/EP0)
      aa = u(np)
      bb = v(np)
      cc = w(np)
      sinpsi = aa*aa + bb*bb
      IF (( sinpsi .GT. 1e-20 )) THEN
        sinpsi = sqrt(sinpsi)
        sindel = bb/sinpsi
        cosdel = aa/sinpsi
      END IF
      IF (( nbr_split .GT. 1 )) THEN
        wt(np) = wt(np)/nbr_split
      END IF
      DO 4051 ibr=1,nbr_split
        IF (( np+1 .GT. 2 )) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,'(//a,i6,a//)') ' Stack overflow in ANNIH! np = ',
     *    np+1, ' Increase $MXSTACK and try again'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
4061    CONTINUE
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          RNNO01 = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          EP=EP0*EXP(RNNO01*WSAMP)
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          RNNO02 = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          REJF = 1 - (EP*A-1)**2/(EP*(A*A-2))
          IF(((RNNO02 .LE. REJF)))GO TO4062
        GO TO 4061
4062    CONTINUE
        ESG1=AVIP*EP
        PESG1=ESG1
        E(NP)=PESG1
        IQ(NP)=0
        IF (( ibr .EQ. 1 )) THEN
          ip = npold
        ELSE
          ip = np-1
        END IF
        X(np)=X(ip)
        Y(np)=Y(ip)
        Z(np)=Z(ip)
        IR(np)=IR(ip)
        WT(np)=WT(ip)
        DNEAR(np)=DNEAR(ip)
        LATCH(np)=LATCH(ip)
        COSTHE=MAX(-1.0,MIN(1.0,(ESG1-RM)*POT/ESG1))
        SINTHE=SQRT(1.0-COSTHE*COSTHE)
4071    CONTINUE
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          xphi = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          xphi = 2*xphi - 1
          xphi2 = xphi*xphi
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          yphi = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          yphi2 = yphi*yphi
          rhophi2 = xphi2 + yphi2
          IF(rhophi2.LE.1)GO TO4072
        GO TO 4071
4072    CONTINUE
        rhophi2 = 1/rhophi2
        cphi = (xphi2 - yphi2)*rhophi2
        sphi = 2*xphi*yphi*rhophi2
        IF (( sinpsi .GE. 1e-10 )) THEN
          us = sinthe*cphi
          vs = sinthe*sphi
          u(np) = cc*cosdel*us - sindel*vs + aa*costhe
          v(np) = cc*sindel*us + cosdel*vs + bb*costhe
          w(np) = cc*costhe - sinpsi*us
        ELSE
          u(np) = sinthe*cphi
          v(np) = sinthe*sphi
          w(np) = cc*costhe
        END IF
        np = np + 1
        PESG2=PAVIP-PESG1
        esg2 = pesg2
        e(np) = pesg2
        iq(np) = 0
        X(np)=X(np-1)
        Y(np)=Y(np-1)
        Z(np)=Z(np-1)
        IR(np)=IR(np-1)
        WT(np)=WT(np-1)
        DNEAR(np)=DNEAR(np-1)
        LATCH(np)=LATCH(np-1)
        COSTHE=MAX(-1.0,MIN(1.0,(ESG2-RM)*POT/ESG2))
        SINTHE=-SQRT(1.0-COSTHE*COSTHE)
        IF (( sinpsi .GE. 1e-10 )) THEN
          us = sinthe*cphi
          vs = sinthe*sphi
          u(np) = cc*cosdel*us - sindel*vs + aa*costhe
          v(np) = cc*sindel*us + cosdel*vs + bb*costhe
          w(np) = cc*costhe - sinpsi*us
        ELSE
          u(np) = sinthe*cphi
          v(np) = sinthe*sphi
          w(np) = cc*costhe
        END IF
        np = np + 1
4051  CONTINUE
4052  CONTINUE
      np = np-1
      RETURN
      END
      SUBROUTINE ANNIH_AT_REST
      implicit none
      COMMON/QDEBUG/QDEBUG
      LOGICAL QDEBUG
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common/egs_vr/ e_max_rr(1),  prob_RR,  nbr_split,  i_play_RR,
     * i_survived_RR,
     *     n_RR_warning,                                        i_do_rr(
     *1)
      real*8          e_max_rr,prob_RR
      integer*4       nbr_split,i_play_RR,i_survived_RR,n_RR_warning
      integer*2     i_do_rr
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      real*8 costhe,sinthe,cphi,sphi
      integer*4 ibr,ip
      real*8 xphi,xphi2,yphi,yphi2,rhophi2
      NPold = NP
      IF (( np+2*nbr_split-1 .GT. 2 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,'(//,3a,/,2(a,i9))') ' In subroutine ','ANNIH_AT_RES
     *T', ' stack size exceeded! ',' $MAXSTACK = ',2,' np = ',np+2*nbr_s
     *  plit-1
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      IF (( nbr_split .GT. 1 )) THEN
        wt(np) = wt(np)/nbr_split
      END IF
      DO 4081 ibr=1,nbr_split
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        costhe = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        costhe = 2*costhe-1
        sinthe = sqrt(max(0.0,(1-costhe)*(1+costhe)))
4091    CONTINUE
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          xphi = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          xphi = 2*xphi - 1
          xphi2 = xphi*xphi
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          yphi = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          yphi2 = yphi*yphi
          rhophi2 = xphi2 + yphi2
          IF(rhophi2.LE.1)GO TO4092
        GO TO 4091
4092    CONTINUE
        rhophi2 = 1/rhophi2
        cphi = (xphi2 - yphi2)*rhophi2
        sphi = 2*xphi*yphi*rhophi2
        e(np) = prm
        iq(np) = 0
        IF (( ibr .EQ. 1 )) THEN
          ip = npold
        ELSE
          ip = np-1
        END IF
        X(np)=X(ip)
        Y(np)=Y(ip)
        Z(np)=Z(ip)
        IR(np)=IR(ip)
        WT(np)=WT(ip)
        DNEAR(np)=DNEAR(ip)
        LATCH(np)=LATCH(ip)
        u(np) = sinthe*cphi
        v(np) = sinthe*sphi
        w(np) = costhe
        np = np+1
        e(np) = prm
        iq(np) = 0
        X(np)=X(np-1)
        Y(np)=Y(np-1)
        Z(np)=Z(np-1)
        IR(np)=IR(np-1)
        WT(np)=WT(np-1)
        DNEAR(np)=DNEAR(np-1)
        LATCH(np)=LATCH(np-1)
        u(np) = -u(np-1)
        v(np) = -v(np-1)
        w(np) = -w(np-1)
        np = np+1
4081  CONTINUE
4082  CONTINUE
      np = np-1
      return
      end
      SUBROUTINE BHABHA
      implicit none
      COMMON/QDEBUG/QDEBUG
      LOGICAL QDEBUG
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/UPHIOT/THETA,SINTHE,COSTHE,SINPHI, COSPHI,PI,TWOPI,PI5D2
      real*8 THETA,  SINTHE,  COSTHE,  SINPHI,  COSPHI,  PI,TWOPI,PI5D2
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      common/egs_vr/ e_max_rr(1),  prob_RR,  nbr_split,  i_play_RR,
     * i_survived_RR,
     *     n_RR_warning,                                        i_do_rr(
     *1)
      real*8          e_max_rr,prob_RR
      integer*4       nbr_split,i_play_RR,i_survived_RR,n_RR_warning
      integer*2     i_do_rr
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      DOUBLE PRECISION PEIP,  PEKIN,  PEKSE2,  PESE1,  PESE2,  H1,  DCOS
     *TH
      real*8 EIP,  EKIN,  T0,  E0,  E02,  YY,  Y2,YP,YP2, BETA2,  EP0,
     *EP0C,  B1,B2,B3,B4,  RNNO03,RNNO04, BR,  REJF2,  ESE1,  ESE2
      NPold = NP
      PEIP=E(NP)
      EIP=PEIP
      PEKIN=PEIP-PRM
      EKIN=PEKIN
      T0=EKIN/RM
      E0=T0+1.
      YY=1./(T0+2.)
      E02=E0*E0
      BETA2=(E02-1.)/E02
      EP0=TE(MEDIUM)/EKIN
      EP0C=1.-EP0
      Y2=YY*YY
      YP=1.-2.*YY
      YP2=YP*YP
      B4=YP2*YP
      B3=B4+YP2
      B2=YP*(3.+Y2)
      B1=2.-Y2
4101  CONTINUE
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        RNNO03 = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        BR=EP0/(1.-EP0C*RNNO03)
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        RNNO04 = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        REJF2=(1.0-BETA2*BR*(B1-BR*(B2-BR*(B3-BR*B4))))
        IF((RNNO04.LE.REJF2))GO TO4102
      GO TO 4101
4102  CONTINUE
      IF (( np+1 .GT. 2 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,'(//,3a,/,2(a,i9))') ' In subroutine ','BHABHA', ' s
     *tack size exceeded! ',' $MAXSTACK = ',2,' np = ',np+1
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      IF ((BR.LT.0.5)) THEN
        IQ(NP+1)=-1
      ELSE
        IQ(NP)=-1
        IQ(NP+1)=1
        BR=1.-BR
      END IF
      BR=max(BR,0.0)
      PEKSE2=BR*EKIN
      PESE1=PEIP-PEKSE2
      PESE2=PEKSE2+PRM
      ESE1=PESE1
      ESE2=PESE2
      E(NP)=PESE1
      E(NP+1)=PESE2
      H1=(PEIP+PRM)/PEKIN
      DCOSTH=MIN(1.0D0,H1*(PESE1-PRM)/(PESE1+PRM))
      SINTHE=DSQRT(1.D0-DCOSTH)
      COSTHE=DSQRT(DCOSTH)
      CALL UPHI(2,1)
      NP=NP+1
      DCOSTH=H1*(PESE2-PRM)/(PESE2+PRM)
      SINTHE=-DSQRT(1.D0-DCOSTH)
      COSTHE=DSQRT(DCOSTH)
      CALL UPHI(3,2)
      RETURN
      END
      SUBROUTINE BREMS
      implicit none
      COMMON/QDEBUG/QDEBUG
      LOGICAL QDEBUG
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      COMMON/EPCONT/EDEP,EDEP_LOCAL,TSTEP,TUSTEP,USTEP,TVSTEP,VSTEP, RHO
     *F,EOLD,ENEW,EKE,ELKE,GLE,E_RANGE, x_final,y_final,z_final, u_final
     *,v_final,w_final, IDISC,IROLD,IRNEW,IAUSFL(35)
      DOUBLE PRECISION EDEP,  EDEP_LOCAL
      real*8 TSTEP,  TUSTEP,  USTEP,  VSTEP,  TVSTEP,  RHOF,  EOLD,  ENE
     *W,  EKE,  ELKE,  GLE,  E_RANGE, x_final,y_final,z_final,  u_final,
     *v_final,w_final
      integer*4 IDISC,  IROLD,  IRNEW,  IAUSFL
      common/nist_brems/ nb_fdata(0:50,100,1), nb_xdata(0:50,100,1), nb_
     *wdata(50,100,1), nb_idata(50,100,1), nb_emin(1),nb_emax(1), nb_lem
     *in(1),nb_lemax(1), nb_dle(1),nb_dlei(1), log_ap(1)
      real*8 nb_fdata,nb_xdata,nb_wdata,nb_emin,nb_emax,nb_lemin,nb_lema
     *x, nb_dle,nb_dlei,log_ap
      integer*4 nb_idata
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/UPHIOT/THETA,SINTHE,COSTHE,SINPHI, COSPHI,PI,TWOPI,PI5D2
      real*8 THETA,  SINTHE,  COSTHE,  SINPHI,  COSPHI,  PI,TWOPI,PI5D2
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      common/egs_vr/ e_max_rr(1),  prob_RR,  nbr_split,  i_play_RR,
     * i_survived_RR,
     *     n_RR_warning,                                        i_do_rr(
     *1)
      real*8          e_max_rr,prob_RR
      integer*4       nbr_split,i_play_RR,i_survived_RR,n_RR_warning
      integer*2     i_do_rr
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      DOUBLE PRECISION PEIE,  PESG,  PESE
      real*8 EIE,  EKIN,  brmin,  waux,  aux,  r1,  ajj,  alias_sample1,
     * RNNO06,  RNNO07,  BR,  ESG,  ESE,  DELTA,  phi1,  phi2,  REJF
      real*8 a,b,c,                               sinpsi, sindel, cosdel
     *, us, vs,
     *                                                ztarg,
     *             tteie,                                    beta,
     *                       y2max,
     *      y2maxi,                                                   tt
     *ese,                                      rjarg1,rjarg2,rjarg3,rej
     *min,rejmid,rejmax,rejtop,rejtst,
     *                 esedei,                                 y2tst,
     *                             y2tst1,
     *                                           rtest,
     *                            xphi,yphi,xphi2,yphi2,rhophi2,cphi,sph
     *i
      integer*4
     *                 L,L1,ibr,jj,j
      real*8 z2max,z2maxi,aux1,aux3,aux4,aux5,aux2,weight
      IF((nbr_split .LT. 1))return
      NPold = NP
      PEIE=E(NP)
      EIE=PEIE
      weight = wt(np)/nbr_split
      IF ((EIE.LT.50.0)) THEN
        L=1
      ELSE
        L=3
      END IF
      L1 = L+1
      ekin = peie-prm
      brmin = ap(medium)/ekin
      waux = elke - log_ap(medium)
      IF (( ibrdst .GE. 0 )) THEN
        a = u(np)
        b = v(np)
        c = w(np)
        sinpsi = a*a + b*b
        IF (( sinpsi .GT. 1e-20 )) THEN
          sinpsi = sqrt(sinpsi)
          sindel = b/sinpsi
          cosdel = a/sinpsi
        END IF
        ztarg = zbrang(medium)
        tteie = eie/rm
        beta = sqrt((tteie-1)*(tteie+1))/tteie
        y2max = 2*beta*(1+beta)*tteie*tteie
        y2maxi = 1/y2max
        IF (( ibrdst .EQ. 1 )) THEN
          z2max = y2max+1
          z2maxi = sqrt(z2max)
        END IF
      END IF
      IF (( ibr_nist .GE. 1 )) THEN
        ajj = 1 + (waux + log_ap(medium) - nb_lemin(medium))*nb_dlei(med
     *  ium)
        jj = ajj
        ajj = ajj - jj
        IF (( jj .GT. 100 )) THEN
          jj = 100
          ajj = -1
        END IF
      END IF
      DO 4111 ibr=1,nbr_split
        IF (( ibr_nist .GE. 1 )) THEN
          IF (( ekin .GT. nb_emin(medium) )) THEN
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            r1 = rng_array(rng_seed)
            rng_seed = rng_seed + 1
            IF (( r1 .LT. ajj )) THEN
              j = jj+1
            ELSE
              j = jj
            END IF
            br = alias_sample1(50,nb_xdata(0,j,medium), nb_fdata(0,j,med
     *      ium), nb_wdata(1,j,medium),nb_idata(1,j,medium))
          ELSE
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            br = rng_array(rng_seed)
            rng_seed = rng_seed + 1
          END IF
          esg = ap(medium)*exp(br*waux)
          pesg = esg
          pese = peie - pesg
          ese = pese
        ELSE
4121      CONTINUE
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            rnno06 = rng_array(rng_seed)
            rng_seed = rng_seed + 1
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            rnno07 = rng_array(rng_seed)
            rng_seed = rng_seed + 1
            br = brmin*exp(rnno06*waux)
            esg = ekin*br
            pesg = esg
            pese = peie - pesg
            ese = pese
            delta = esg/eie/ese*delcm(medium)
            aux = ese/eie
            IF (( delta .LT. 1 )) THEN
              phi1 = dl1(l,medium)+delta*(dl2(l,medium)+delta*dl3(l,medi
     *        um))
              phi2 = dl1(l1,medium)+delta*(dl2(l1,medium)+ delta*dl3(l1,
     *        medium))
            ELSE
              phi1 = dl4(l,medium)+dl5(l,medium)*log(delta+dl6(l,medium)
     *        )
              phi2 = phi1
            END IF
            rejf = (1+aux*aux)*phi1 - 2*aux*phi2/3
            IF(((rnno07 .LT. rejf)))GO TO4122
          GO TO 4121
4122      CONTINUE
        END IF
        np=np+1
        IF (( np .GT. 2 )) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,'(//a,i6,a//)') ' Stack overflow in BREMS! np = ',
     *    np+1, ' Increase $MXSTACK and try again'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        e(np) = pesg
        iq(np) = 0
        X(np)=X(np-1)
        Y(np)=Y(np-1)
        Z(np)=Z(np-1)
        IR(np)=IR(np-1)
        WT(np)=WT(np-1)
        DNEAR(np)=DNEAR(np-1)
        LATCH(np)=LATCH(np-1)
        wt(np) = weight
        IF (( ibrdst .LT. 0 )) THEN
          u(np) = u(npold)
          v(np) = v(npold)
          w(np) = w(npold)
        ELSE
          IF (( ibrdst .EQ. 1 )) THEN
            ttese = ese/rm
            esedei = ttese/tteie
            rjarg1 = 1+esedei*esedei
            rjarg2 = rjarg1 + 2*esedei
            aux = 2*ese*tteie/esg
            aux = aux*aux
            aux1 = aux*ztarg
            IF (( aux1 .GT. 10 )) THEN
              rjarg3 = lzbrang(medium) + (1-aux1)/aux1**2
            ELSE
              rjarg3 = log(aux/(1+aux1))
            END IF
            rejmax = rjarg1*rjarg3-rjarg2
4131        CONTINUE
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              y2tst = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              rtest = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              aux3 = z2maxi/(y2tst+(1-y2tst)*z2maxi)
              rtest = rtest*aux3*rejmax
              y2tst = aux3**2-1
              y2tst1 = esedei*y2tst/aux3**4
              aux4 = 16*y2tst1-rjarg2
              aux5 = rjarg1-4*y2tst1
              IF((rtest .LT. aux4 + aux5*rjarg3))GO TO4132
              aux2 = log(aux/(1+aux1/aux3**4))
              rejtst = aux4+aux5*aux2
              IF(((rtest .LT. rejtst )))GO TO4132
            GO TO 4131
4132        CONTINUE
          ELSE
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            y2tst = rng_array(rng_seed)
            rng_seed = rng_seed + 1
            y2tst = y2tst/(1-y2tst+y2maxi)
          END IF
          costhe = 1 - 2*y2tst*y2maxi
          sinthe = sqrt(max((1-costhe)*(1+costhe),0.0))
4141      CONTINUE
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            xphi = rng_array(rng_seed)
            rng_seed = rng_seed + 1
            xphi = 2*xphi - 1
            xphi2 = xphi*xphi
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            yphi = rng_array(rng_seed)
            rng_seed = rng_seed + 1
            yphi2 = yphi*yphi
            rhophi2 = xphi2 + yphi2
            IF(rhophi2.LE.1)GO TO4142
          GO TO 4141
4142      CONTINUE
          rhophi2 = 1/rhophi2
          cphi = (xphi2 - yphi2)*rhophi2
          sphi = 2*xphi*yphi*rhophi2
          IF (( sinpsi .GE. 1e-10 )) THEN
            us = sinthe*cphi
            vs = sinthe*sphi
            u(np) = c*cosdel*us - sindel*vs + a*costhe
            v(np) = c*sindel*us + cosdel*vs + b*costhe
            w(np) = c*costhe - sinpsi*us
          ELSE
            u(np) = sinthe*cphi
            v(np) = sinthe*sphi
            w(np) = c*costhe
          END IF
        END IF
4111  CONTINUE
4112  CONTINUE
      e(npold) = pese
      RETURN
      END
      SUBROUTINE COMPT
      implicit none
      common/compton_data/ iz_array(1538),  be_array(1538),  Jo_array(15
     *38),  erfJo_array(1538),   ne_array(1538),  shn_array(1538),
     *shell_array(200,1), eno_array(200,1), eno_atbin_array(200,1), n_sh
     *ell(1), radc_flag,  ibcmp(1)
      integer*4 iz_array,ne_array,shn_array,eno_atbin_array, shell_array
     *,n_shell,radc_flag
      real*8 be_array,Jo_array,erfJo_array,eno_array
      integer*2 ibcmp
      COMMON/QDEBUG/QDEBUG
      LOGICAL QDEBUG
      COMMON/EPCONT/EDEP,EDEP_LOCAL,TSTEP,TUSTEP,USTEP,TVSTEP,VSTEP, RHO
     *F,EOLD,ENEW,EKE,ELKE,GLE,E_RANGE, x_final,y_final,z_final, u_final
     *,v_final,w_final, IDISC,IROLD,IRNEW,IAUSFL(35)
      DOUBLE PRECISION EDEP,  EDEP_LOCAL
      real*8 TSTEP,  TUSTEP,  USTEP,  VSTEP,  TVSTEP,  RHOF,  EOLD,  ENE
     *W,  EKE,  ELKE,  GLE,  E_RANGE, x_final,y_final,z_final,  u_final,
     *v_final,w_final
      integer*4 IDISC,  IROLD,  IRNEW,  IAUSFL
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/UPHIOT/THETA,SINTHE,COSTHE,SINPHI, COSPHI,PI,TWOPI,PI5D2
      real*8 THETA,  SINTHE,  COSTHE,  SINPHI,  COSPHI,  PI,TWOPI,PI5D2
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      common/egs_vr/ e_max_rr(1),  prob_RR,  nbr_split,  i_play_RR,
     * i_survived_RR,
     *     n_RR_warning,                                        i_do_rr(
     *1)
      real*8          e_max_rr,prob_RR
      integer*4       nbr_split,i_play_RR,i_survived_RR,n_RR_warning
      integer*2     i_do_rr
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      common/relax_data/ relax_first(3000),  relax_ntran(3000),  relax_s
     *tate(10000),  relax_prob(10000),  relax_atbin(10000),  relax_ntot
      real*8 relax_prob
      integer*4 relax_first, relax_ntran, relax_state, relax_atbin, rela
     *x_ntot
      DOUBLE PRECISION PEIG,  PESG,  PESE
      real*8 ko,  broi,  broi2,  bro,  bro1,  alph1,  alph2,  alpha,  rn
     *no15,rnno16,rnno17,rnno18,rnno19,  br,  temp,  rejf3,  rejmax,  Uj
     *,  Jo,  br2,  fpz,fpz1, qc,  qc2,  af,  Fmax,  frej,  eta_incoh, e
     *ta,  aux,aux1,aux2,aux3,aux4,  pzmax,  pz,  pz2,  rnno_RR
      integer*4 irl,  i,  j,  iarg,  ip
      logical first_time
      integer*4 ibcmpl
      NPold = NP
      peig=E(NP)
      ko = peig/rm
      broi = 1 + 2*ko
      irl = ir(np)
      first_time = .true.
      ibcmpl = ibcmp(irl)
4150  CONTINUE
      IF (( ibcmpl .GT. 0 )) THEN
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        rnno17 = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        rnno17 = 1 + rnno17*n_shell(medium)
        i = int(rnno17)
        IF((rnno17 .GT. eno_array(i,medium)))i = eno_atbin_array(i,mediu
     *  m)
        j = shell_array(i,medium)
        Uj = be_array(j)
        IF (( ko .LE. Uj )) THEN
          IF (( ibcmpl .EQ. 1 )) THEN
            goto 4160
          ELSE
            goto 4150
          END IF
        END IF
        Jo = Jo_array(j)
      END IF
4170  CONTINUE
      IF (( ko .GT. 2 )) THEN
        IF (( first_time )) THEN
          broi2 = broi*broi
          alph1 = Log(broi)
          bro = 1/broi
          alph2 = ko*(broi+1)*bro*bro
          alpha = alph1+alph2
        END IF
4181    CONTINUE
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          rnno15 = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          rnno16 = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          IF (( rnno15*alpha .LT. alph1 )) THEN
            br = Exp(alph1*rnno16)*bro
          ELSE
            br = Sqrt(rnno16*broi2 + (1-rnno16))*bro
          END IF
          temp = (1-br)/(ko*br)
          sinthe = Max(0.,temp*(2-temp))
          aux = 1+br*br
          rejf3 = aux - br*sinthe
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          rnno19 = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          IF((rnno19*aux.le.rejf3))GO TO4182
        GO TO 4181
4182    CONTINUE
      ELSE
        IF (( first_time )) THEN
          bro = 1./broi
          bro1 = 1 - bro
          rejmax = broi + bro
        END IF
4191    CONTINUE
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          rnno15 = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          rnno16 = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          br = bro + bro1*rnno15
          temp = (1-br)/(ko*br)
          sinthe = Max(0.,temp*(2-temp))
          rejf3 = 1 + br*br - br*sinthe
          IF((rnno16*br*rejmax.le.rejf3))GO TO4192
        GO TO 4191
4192    CONTINUE
      END IF
      first_time = .false.
      IF ((br .LT. bro .OR. br .GT. 1)) THEN
        IF (( br .LT. 0.99999/broi .OR. br .GT. 1.00001 )) THEN
          write(i_log,'(/a)') '***************** Warning: '
          write(i_log,*) ' sampled br outside of allowed range! ',ko,1./
     *    broi,br
        END IF
        goto 4170
      END IF
      costhe = 1 - temp
      IF (( ibcmp(irl) .EQ. 0 )) THEN
        Uj = 0
        goto 4200
      END IF
      br2 = br*br
      aux = ko*(ko-Uj)*temp
      aux1 = 2*aux + Uj*Uj
      pzmax = aux - Uj
      IF (( pzmax .LT. 0 .AND. pzmax*pzmax .GE. aux1 )) THEN
        IF (( ibcmpl .EQ. 1 )) THEN
          goto 4160
        ELSE
          goto 4150
        END IF
      END IF
      pzmax = pzmax/sqrt(aux1)
      qc2 = 1 + br*br - 2*br*costhe
      qc = sqrt(qc2)
      IF (( pzmax .GT. 1 )) THEN
        pzmax = 1
        af = 0
        Fmax = 1
        fpz = 1
        goto 4210
      END IF
      aux3 = 1 + 2*Jo*abs(pzmax)
      aux4 = 0.5*(1-aux3*aux3)
      fpz = 0.5*exp(aux4)
      af = qc*(1+br*(br-costhe)/qc2)
      IF (( af .LT. 0 )) THEN
        IF((pzmax .GT. 0))fpz = 1 - fpz
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        eta_incoh = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        IF (( eta_incoh .GT. fpz )) THEN
          IF (( ibcmpl .EQ. 1 )) THEN
            goto 4160
          ELSE
            goto 4150
          END IF
        END IF
        af = 0
        Fmax = 1
        goto 4210
      END IF
      IF (( pzmax .LT. -0.15 )) THEN
        Fmax = 1-af*0.15
        fpz1 = fpz*Fmax*Jo
      ELSE IF(( pzmax .LT. 0.15 )) THEN
        Fmax = 1 + af*pzmax
        aux3 = 1/(1+0.33267252734*aux3)
        aux4 = fpz*aux3*(0.3480242+aux3*(-0.0958798+aux3*0.7478556)) + e
     *  rfJo_array(j)
        IF (( pzmax .GT. 0 )) THEN
          fpz1 = (1 - Fmax*fpz)*Jo - 0.62665706866*af*aux4
          fpz = 1 - fpz
        ELSE
          fpz1 = Fmax*fpz*Jo - 0.62665706866*af*aux4
        END IF
      ELSE
        Fmax = 1 + af*0.15
        fpz1 = (1 - Fmax*fpz)*Jo
        fpz = 1 - fpz
      END IF
      IF (( rng_seed .GT. 24 )) THEN
        call ranlux(rng_array)
        rng_seed = 1
      END IF
      eta_incoh = rng_array(rng_seed)
      rng_seed = rng_seed + 1
      IF ((eta_incoh*Jo .GT. fpz1 )) THEN
        IF (( ibcmpl .EQ. 1 )) THEN
          goto 4160
        ELSE
          goto 4150
        END IF
      END IF
4210  CONTINUE
      IF (( ibcmpl .NE. 2 )) THEN
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        rnno18 = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        rnno18 = rnno18*fpz
        IF (( rnno18 .LT. 0.5 )) THEN
          rnno18 = Max(1e-30,2*rnno18)
          pz = 0.5*(1-Sqrt(1-2*Log(rnno18)))/Jo
        ELSE
          rnno18 = 2*(1-rnno18)
          pz = 0.5*(Sqrt(1-2*Log(rnno18))-1)/Jo
        END IF
        IF((abs(pz) .GT. 1))goto 4210
        IF (( pz .LT. 0.15 )) THEN
          IF (( pz .LT. -0.15 )) THEN
            frej = 1 - af*0.15
          ELSE
            frej = 1 + af*pz
          END IF
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          eta = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          IF((eta*Fmax .GT. frej))goto 4210
        END IF
      ELSE
        pz = 0
        Uj = 0
      END IF
      pz2 = pz*pz
      IF (( abs(pz) .LT. 0.01 )) THEN
        br = br*(1 + pz*(qc + (br2-costhe)*pz))
      ELSE
        aux = 1 - pz2*br*costhe
        aux1 = 1 - pz2*br2
        aux2 = qc2 - br2*pz2*sinthe
        IF (( aux2 .GT. 1e-10 )) THEN
          br = br/aux1*(aux+pz*Sqrt(aux2))
        END IF
      END IF
      Uj = Uj*prm
4200  pesg = br*peig
      pese = peig - pesg - Uj + prm
      sinthe = Sqrt(sinthe)
      call uphi(2,1)
      e(np) = pesg
      aux = 1 + br*br - 2*br*costhe
      IF (( aux .GT. 1e-8 )) THEN
        costhe = (1-br*costhe)/Sqrt(aux)
        sinthe = (1-costhe)*(1+costhe)
        IF (( sinthe .GT. 0 )) THEN
          sinthe = -Sqrt(sinthe)
        ELSE
          sinthe = 0
        END IF
      ELSE
        costhe = 0
        sinthe = -1
      END IF
      np = np + 1
      IF (( np .GT. 2 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,'(//,3a,/,2(a,i9))') ' In subroutine ','COMPT', ' st
     *ack size exceeded! ',' $MAXSTACK = ',2,' np = ',np
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      call uphi(3,2)
      e(np) = pese
      iq(np) = -1
      IF (( ibcmpl .EQ. 1 .OR. ibcmpl .EQ. 3 )) THEN
        IF (( Uj .GT. 1e-3 )) THEN
          edep = pzero
          call relax(Uj,shn_array(j),iz_array(j))
        ELSE
          edep = Uj
          edep_local = edep
          IARG=33
          IF ((IAUSFL(IARG+1).NE.0)) THEN
            CALL AUSGAB(IARG)
          END IF
        END IF
        IF (( edep .GT. 0 )) THEN
          IARG=4
          IF ((IAUSFL(IARG+1).NE.0)) THEN
            CALL AUSGAB(IARG)
          END IF
        END IF
      END IF
      i_survived_RR = 0
      IF (( i_play_RR .EQ. 1 )) THEN
        IF (( prob_RR .LE. 0 )) THEN
          IF (( n_RR_warning .LT. 50 )) THEN
            n_RR_warning = n_RR_warning + 1
            WRITE(6,4220)prob_RR
4220        FORMAT('**** Warning, attempt to play Roussian Roulette with
     * prob_RR<=0! ',g14.6)
          END IF
        ELSE
          ip = NPold+1
4231      CONTINUE
            IF (( iq(ip) .NE. 0 )) THEN
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              rnno_RR = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              IF (( rnno_RR .LT. prob_RR )) THEN
                wt(ip) = wt(ip)/prob_RR
                ip = ip + 1
              ELSE
                i_survived_RR = i_survived_RR + 1
                IF ((ip .LT. np)) THEN
                  e(ip) = e(np)
                  iq(ip) = iq(np)
                  wt(ip) = wt(np)
                  u(ip) = u(np)
                  v(ip) = v(np)
                  w(ip) = w(np)
                END IF
                np = np-1
              END IF
            ELSE
              ip = ip+1
            END IF
            IF(((ip .GT. np)))GO TO4232
          GO TO 4231
4232      CONTINUE
          IF (( np .EQ. 0 )) THEN
            np = 1
            e(np) = 0
            iq(np) = 0
            wt(np) = 0
          END IF
        END IF
      END IF
      return
4160  return
      end
      SUBROUTINE old_COMPT
      implicit none
      common/compton_data/ iz_array(1538),  be_array(1538),  Jo_array(15
     *38),  erfJo_array(1538),   ne_array(1538),  shn_array(1538),
     *shell_array(200,1), eno_array(200,1), eno_atbin_array(200,1), n_sh
     *ell(1), radc_flag,  ibcmp(1)
      integer*4 iz_array,ne_array,shn_array,eno_atbin_array, shell_array
     *,n_shell,radc_flag
      real*8 be_array,Jo_array,erfJo_array,eno_array
      integer*2 ibcmp
      COMMON/QDEBUG/QDEBUG
      LOGICAL QDEBUG
      COMMON/EPCONT/EDEP,EDEP_LOCAL,TSTEP,TUSTEP,USTEP,TVSTEP,VSTEP, RHO
     *F,EOLD,ENEW,EKE,ELKE,GLE,E_RANGE, x_final,y_final,z_final, u_final
     *,v_final,w_final, IDISC,IROLD,IRNEW,IAUSFL(35)
      DOUBLE PRECISION EDEP,  EDEP_LOCAL
      real*8 TSTEP,  TUSTEP,  USTEP,  VSTEP,  TVSTEP,  RHOF,  EOLD,  ENE
     *W,  EKE,  ELKE,  GLE,  E_RANGE, x_final,y_final,z_final,  u_final,
     *v_final,w_final
      integer*4 IDISC,  IROLD,  IRNEW,  IAUSFL
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/UPHIOT/THETA,SINTHE,COSTHE,SINPHI, COSPHI,PI,TWOPI,PI5D2
      real*8 THETA,  SINTHE,  COSTHE,  SINPHI,  COSPHI,  PI,TWOPI,PI5D2
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      common/egs_vr/ e_max_rr(1),  prob_RR,  nbr_split,  i_play_RR,
     * i_survived_RR,
     *     n_RR_warning,                                        i_do_rr(
     *1)
      real*8          e_max_rr,prob_RR
      integer*4       nbr_split,i_play_RR,i_survived_RR,n_RR_warning
      integer*2     i_do_rr
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      common/relax_data/ relax_first(3000),  relax_ntran(3000),  relax_s
     *tate(10000),  relax_prob(10000),  relax_atbin(10000),  relax_ntot
      real*8 relax_prob
      integer*4 relax_first, relax_ntran, relax_state, relax_atbin, rela
     *x_ntot
      DOUBLE PRECISION PEIG,  PESG,  PESE
      real*8 ko,  broi,  broi2,  bro,  bro1,  alph1,  alph2,  alpha,  rn
     *no15,rnno16,rnno17,rnno18,rnno19,  br,  temp,  rejf3,  rejmax,  Uj
     *,  br2,  aux,aux1,aux2, pzmax2,  pz,  pz2,  rnno_RR
      integer*4 irl,  i,  j,  iarg,  ip
      i_survived_RR = 0
      NPold = NP
      peig=E(NP)
      ko = peig/rm
      broi = 1 + 2*ko
      irl = ir(np)
      IF (( ibcmp(irl) .EQ. 1 )) THEN
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        rnno17 = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        DO 4241 i=1,n_shell(medium)
          rnno17 = rnno17 - eno_array(i,medium)
          IF((rnno17 .LE. 0))GO TO4242
4241    CONTINUE
4242    CONTINUE
        j = shell_array(i,medium)
        Uj = be_array(j)
        IF (( ko .LE. Uj )) THEN
          goto 4250
        END IF
      END IF
4260  CONTINUE
      IF (( ko .GT. 2 )) THEN
        broi2 = broi*broi
        alph1 = Log(broi)
        alph2 = ko*(broi+1)/broi2
        alpha = alph1/(alph1+alph2)
4271    CONTINUE
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          rnno15 = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          rnno16 = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          IF (( rnno15 .LT. alpha )) THEN
            br = Exp(alph1*rnno16)/broi
          ELSE
            br = Sqrt(rnno16 + (1-rnno16)/broi2)
          END IF
          temp = (1-br)/ko/br
          sinthe = Max(0.,temp*(2-temp))
          rejf3 = 1 - br*sinthe/(1+br*br)
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          rnno19 = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          IF((rnno19.le.rejf3))GO TO4272
        GO TO 4271
4272    CONTINUE
      ELSE
        bro = 1./broi
        bro1 = 1 - bro
        rejmax = broi + bro
4281    CONTINUE
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          rnno15 = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          rnno16 = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          br = bro + bro1*rnno15
          temp = (1-br)/ko/br
          sinthe = Max(0.,temp*(2-temp))
          rejf3 = (br + 1./br - sinthe)/rejmax
          IF((rnno16.le.rejf3))GO TO4282
        GO TO 4281
4282    CONTINUE
      END IF
      IF ((br .LT. 1./broi .OR. br .GT. 1)) THEN
        IF (( br .LT. 0.99999/broi .OR. br .GT. 1.00001 )) THEN
          write(i_log,'(/a)') '***************** Warning: '
          write(i_log,*) ' sampled br outside of allowed range! ',ko,1./
     *    broi,br
        END IF
        goto 4260
      END IF
      IF (( ibcmp(irl) .EQ. 0 )) THEN
        Uj = 0
        costhe = 1 - temp
        goto 4290
      END IF
      br2 = br*br
      costhe = 1 - temp
      aux = ko*(ko-Uj)*temp
      aux1 = aux-Uj
      pzmax2 = aux1*aux1/(2*aux+Uj*Uj)
4300  CONTINUE
      IF (( rng_seed .GT. 24 )) THEN
        call ranlux(rng_array)
        rng_seed = 1
      END IF
      rnno18 = rng_array(rng_seed)
      rng_seed = rng_seed + 1
      IF (( rnno18 .LT. 0.5 )) THEN
        rnno18 = Max(1e-30,2*rnno18)
        pz = 0.5*(1-Sqrt(1-2*Log(rnno18)))/Jo_array(j)
        pz2 = pz*pz
        IF (( (pz2 .LE. pzmax2) .AND. (aux1 .LT. 0) )) THEN
          goto 4250
        END IF
      ELSE
        IF (( aux1 .LT. 0 )) THEN
          goto 4250
        END IF
        rnno18 = 2*(1-rnno18)
        pz = 0.5*(Sqrt(1-2*Log(rnno18))-1)/Jo_array(j)
        pz2 = pz*pz
        IF (( pz2 .GE. pzmax2 )) THEN
          goto 4250
        END IF
      END IF
      IF((abs(pz) .GT. 1))goto 4300
      aux = 1 - pz2*br*costhe
      aux1 = 1 - pz2*br2
      aux2 = 1-2*br*costhe+br2*(1-pz2*sinthe)
      IF (( aux2 .GT. 1e-10 )) THEN
        br = br/aux1*(aux+pz*Sqrt(aux2))
      END IF
      Uj = Uj*prm
4290  pesg = br*peig
      pese = peig - pesg - Uj + prm
      sinthe = Sqrt(sinthe)
      call uphi(2,1)
      e(np) = pesg
      aux = 1 + br*br - 2*br*costhe
      IF (( aux .GT. 1e-8 )) THEN
        costhe = (1-br*costhe)/Sqrt(aux)
        sinthe = (1-costhe)*(1+costhe)
        IF (( sinthe .GT. 0 )) THEN
          sinthe = -Sqrt(sinthe)
        ELSE
          sinthe = 0
        END IF
      ELSE
        costhe = 0
        sinthe = -1
      END IF
      np = np + 1
      IF (( np .GT. 2 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,'(//,3a,/,2(a,i9))') ' In subroutine ','COMPT', ' st
     *ack size exceeded! ',' $MAXSTACK = ',2,' np = ',np
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      call uphi(3,2)
      e(np) = pese
      iq(np) = -1
      IF (( ibcmp(irl) .EQ. 1 )) THEN
        IF (( Uj .GT. 1e-3 )) THEN
          edep = 0
          call relax(Uj,shn_array(j),iz_array(j))
        ELSE
          edep = Uj
        END IF
        IF (( edep .GT. 0 )) THEN
          IARG=4
          IF ((IAUSFL(IARG+1).NE.0)) THEN
            CALL AUSGAB(IARG)
          END IF
        END IF
      END IF
      i_survived_RR = 0
      IF (( i_play_RR .EQ. 1 )) THEN
        IF (( prob_RR .LE. 0 )) THEN
          IF (( n_RR_warning .LT. 50 )) THEN
            n_RR_warning = n_RR_warning + 1
            WRITE(6,4310)prob_RR
4310        FORMAT('**** Warning, attempt to play Roussian Roulette with
     * prob_RR<=0! ',g14.6)
          END IF
        ELSE
          ip = NPold+1
4321      CONTINUE
            IF (( iq(ip) .NE. 0 )) THEN
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              rnno_RR = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              IF (( rnno_RR .LT. prob_RR )) THEN
                wt(ip) = wt(ip)/prob_RR
                ip = ip + 1
              ELSE
                i_survived_RR = i_survived_RR + 1
                IF ((ip .LT. np)) THEN
                  e(ip) = e(np)
                  iq(ip) = iq(np)
                  wt(ip) = wt(np)
                  u(ip) = u(np)
                  v(ip) = v(np)
                  w(ip) = w(np)
                END IF
                np = np-1
              END IF
            ELSE
              ip = ip+1
            END IF
            IF(((ip .GT. np)))GO TO4322
          GO TO 4321
4322      CONTINUE
          IF (( np .EQ. 0 )) THEN
            np = 1
            e(np) = 0
            iq(np) = 0
            wt(np) = 0
          END IF
        END IF
      END IF
      return
4250  return
      end
      SUBROUTINE ELECTR(IRCODE)
      implicit none
      integer*4 IRCODE
      COMMON/QDEBUG/QDEBUG
      LOGICAL QDEBUG
      COMMON/BOUNDS/ECUT(1),PCUT(1),VACDST
      real*8 ECUT,  PCUT,  VACDST
      COMMON/ELECIN/ esig_e(1),psig_e(1), esige_max, psige_max, range_ep
     *(0:1,500,1), E_array(500,1), etae_ms0(500,1),etae_ms1(500,1),etap_
     *ms0(500,1),etap_ms1(500,1),q1ce_ms0(500,1),q1ce_ms1(500,1),q1cp_ms
     *0(500,1),q1cp_ms1(500,1),q2ce_ms0(500,1),q2ce_ms1(500,1),q2cp_ms0(
     *500,1),q2cp_ms1(500,1),blcce0(500,1),blcce1(500,1), EKE0(1),EKE1(1
     *), XR0(1),TEFF0(1),BLCC(1),XCC(1), ESIG0(500,1),ESIG1(500,1),PSIG0
     *(500,1),PSIG1(500,1),EDEDX0(500,1),EDEDX1(500,1),PDEDX0(500,1),PDE
     *DX1(500,1),EBR10(500,1),EBR11(500,1),PBR10(500,1),PBR11(500,1),PBR
     *20(500,1),PBR21(500,1),TMXS0(500,1),TMXS1(500,1), expeke1(1), IUNR
     *ST(1),EPSTFL(1),IAPRIM(1), sig_ismonotone(0:1,1)
      real*8 esig_e,   psig_e,   esige_max,  psige_max,  range_ep,  E_ar
     *ray,  etae_ms0,etae_ms1,  etap_ms0,etap_ms1,  q1ce_ms0,q1ce_ms1,
     *q1cp_ms0,q1cp_ms1,  q2ce_ms0,q2ce_ms1,  q2cp_ms0,q2cp_ms1,  blcce0
     *,blcce1,   expeke1,  EKE0,EKE1, XR0,  TEFF0,  BLCC,  XCC,  ESIG0,E
     *SIG1,  PSIG0,PSIG1,  EDEDX0,EDEDX1,  PDEDX0,PDEDX1,  EBR10,EBR11,
     * PBR10,PBR11,  PBR20,PBR21,  TMXS0,TMXS1
      integer*4 IUNRST,  EPSTFL,  IAPRIM
      logical sig_ismonotone
      COMMON/EPCONT/EDEP,EDEP_LOCAL,TSTEP,TUSTEP,USTEP,TVSTEP,VSTEP, RHO
     *F,EOLD,ENEW,EKE,ELKE,GLE,E_RANGE, x_final,y_final,z_final, u_final
     *,v_final,w_final, IDISC,IROLD,IRNEW,IAUSFL(35)
      DOUBLE PRECISION EDEP,  EDEP_LOCAL
      real*8 TSTEP,  TUSTEP,  USTEP,  VSTEP,  TVSTEP,  RHOF,  EOLD,  ENE
     *W,  EKE,  ELKE,  GLE,  E_RANGE, x_final,y_final,z_final,  u_final,
     *v_final,w_final
      integer*4 IDISC,  IROLD,  IRNEW,  IAUSFL
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/MISC/  DUNIT,KMPI,KMPO,RHOR(1),MED(1),IRAYLR(1),IPHOTONUCR(
     *1)
      real*8 DUNIT,  RHOR
      integer*4 KMPI,  KMPO
      integer*2 MED,  IRAYLR,  IPHOTONUCR
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/UPHIIN/SINC0,SINC1,SIN0(1002),SIN1(1002)
      real*8 SINC0,SINC1,SIN0,SIN1
      COMMON/UPHIOT/THETA,SINTHE,COSTHE,SINPHI, COSPHI,PI,TWOPI,PI5D2
      real*8 THETA,  SINTHE,  COSTHE,  SINPHI,  COSPHI,  PI,TWOPI,PI5D2
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      common/ET_control/ smaxir(1),estepe,ximax,  skindepth_for_bca,tran
     *sport_algorithm, bca_algorithm,exact_bca,spin_effects
      real*8 smaxir,  estepe,  ximax,      skindepth_for_bca
      integer*4 transport_algorithm, bca_algorithm
      logical exact_bca,  spin_effects
      common/CH_steps/ count_pII_steps,count_all_steps,is_ch_step
      real*8 count_pII_steps,count_all_steps
      logical is_ch_step
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      common/egs_vr/ e_max_rr(1),  prob_RR,  nbr_split,  i_play_RR,
     * i_survived_RR,
     *     n_RR_warning,                                        i_do_rr(
     *1)
      real*8          e_max_rr,prob_RR
      integer*4       nbr_split,i_play_RR,i_survived_RR,n_RR_warning
      integer*2     i_do_rr
      common/emf_inputs/ExIN,EyIN,EzIN,  EMLMTIN,  BxIN, ByIN, BzIN,  Bx
     *, By, Bz,  Bx_new, By_new, Bz_new,  emfield_on
      real*8 ExIN,EyIN,EzIN, EMLMTIN, BxIN,ByIN,BzIN, Bx,By,Bz, Bx_new,B
     *y_new,Bz_new
      logical emfield_on
      common/eii_data/ eii_xsection_a( 10000),  eii_xsection_b( 10000),
     * eii_cons(1), eii_a(40),  eii_b(40),  eii_L_factor,  eii_z(40),  e
     *ii_sh(40),  eii_nshells(100),  eii_nsh(1),  eii_first(1,50),  eii_
     *no(1,50),  eii_flag
      real*8 eii_xsection_a,eii_xsection_b,eii_a,eii_b,eii_cons,eii_L_fa
     *ctor
      integer*4 eii_z,eii_sh,eii_nshells
      integer*4 eii_first,eii_no
      integer*4 eii_elements,eii_flag,eii_nsh
      real*8 lambda_max, sigratio, u_tmp, v_tmp, w_tmp
      LOGICAL random_tustep
      DOUBLE PRECISION  demfp,  peie,  total_tstep,  total_de
      real*8 ekems,  elkems,  chia2,  etap,  lambda,  blccl,  xccl,  xi,
     *  xi_corr,  ms_corr, p2,  beta2,  de,  save_de,  dedx,  dedx0,  de
     *dxmid,  ekei,  elkei,  aux,  ebr1,  eie,  ekef,  elkef,  ekeold,
     *eketmp,  elktmp,  fedep,  tuss,  pbr1,  pbr2,  range,  rfict,  rnn
     *e1,  rnno24,  rnno25,  rnnotu,  rnnoss,  sig,  sig0,  sigf,  skind
     *epth,  ssmfp,  tmxs,  tperp,  ustep0,  uscat,  vscat,  wscat,  xtr
     *ans,  ytrans,  ztrans,  cphi,sphi
      real*8 xphi,xphi2,yphi,yphi2,rhophi2
      integer*4 iarg,  idr,  ierust,  irl,  lelec,  qel,  lelke,  lelkem
     *s,  lelkef,  lelktmp,  ibr
      logical  callhowfar,   domultiple,  dosingle,   callmsdist,
     *                findindex,
     *              spin_index,                                   comput
     *e_tstep
     *
      data ierust/0/
      save ierust
      ircode = 1
      irold = ir(np)
      irl = irold
      medium = med(irl)
4330  CONTINUE
4331    CONTINUE
        lelec = iq(np)
        qel = (1+lelec)/2
        peie = e(np)
        eie = peie
        IF ((eie .LE. ecut(irl))) THEN
          go to 4340
        END IF
        IF ((WT(NP) .EQ. 0.0)) THEN
          go to 4350
        END IF
4360    CONTINUE
4361      CONTINUE
          compute_tstep = .true.
          eke = eie - rm
          IF ((medium .NE. 0)) THEN
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            RNNE1 = rng_array(rng_seed)
            rng_seed = rng_seed + 1
            IF ((RNNE1.EQ.0.0)) THEN
              RNNE1=1.E-30
            END IF
            DEMFP=MAX(-LOG(RNNE1),1.E-8)
            elke = log(eke)
            Lelke=eke1(MEDIUM)*elke+eke0(MEDIUM)
            IF (( sig_ismonotone(qel,medium) )) THEN
              IF ((lelec .LT. 0)) THEN
                sigf=esig1(Lelke,MEDIUM)*elke+esig0(Lelke,MEDIUM)
                dedx0=ededx1(Lelke,MEDIUM)*elke+ededx0(Lelke,MEDIUM)
                sigf = sigf/dedx0
              ELSE
                sigf=psig1(Lelke,MEDIUM)*elke+psig0(Lelke,MEDIUM)
                dedx0=pdedx1(Lelke,MEDIUM)*elke+pdedx0(Lelke,MEDIUM)
                sigf = sigf/dedx0
              END IF
              sig0 = sigf
            ELSE
              IF (( lelec .LT. 0 )) THEN
                sig0 = esig_e(medium)
              ELSE
                sig0 = psig_e(medium)
              END IF
            END IF
          END IF
4370      CONTINUE
4371        CONTINUE
            IF ((medium .EQ. 0)) THEN
              tstep = vacdst
              ustep = tstep
              tustep = ustep
              callhowfar = .true.
              ustep = tustep
            ELSE
              RHOF=RHOR(IRL)/RHO(MEDIUM)
              sig = sig0
              IF ((sig .LE. 0)) THEN
                tstep = vacdst
                sig0 = 1.E-15
              ELSE
                IF (( compute_tstep )) THEN
                  total_de = demfp/sig
                  fedep = total_de
                  ekef = eke - fedep
                  IF (( ekef .LE. E_array(1,medium) )) THEN
                    tstep = vacdst
                  ELSE
                    elkef = Log(ekef)
                    Lelkef=eke1(MEDIUM)*elkef+eke0(MEDIUM)
                    IF (( lelkef .EQ. lelke )) THEN
                      fedep = 1 - ekef/eke
                      elktmp = 0.5*(elke+elkef+0.25*fedep*fedep*(1+fedep
     *                *(1+0.875*fedep)))
                      lelktmp = lelke
                      IF ((lelec .LT. 0)) THEN
                        dedxmid=ededx1(Lelktmp,MEDIUM)*elktmp+ededx0(Lel
     *                  ktmp,MEDIUM)
                        dedxmid = 1/dedxmid
                        aux = ededx1(lelktmp,medium)*dedxmid
                      ELSE
                        dedxmid=pdedx1(Lelktmp,MEDIUM)*elktmp+pdedx0(Lel
     *                  ktmp,MEDIUM)
                        dedxmid = 1/dedxmid
                        aux = pdedx1(lelktmp,medium)*dedxmid
                      END IF
                      aux = aux*(1+2*aux)*(fedep/(2-fedep))**2/6
                      tstep = fedep*eke*dedxmid*(1+aux)
                    ELSE
                      ekei = E_array(lelke,medium)
                      elkei = (lelke - eke0(medium))/eke1(medium)
                      fedep = 1 - ekei/eke
                      elktmp = 0.5*(elke+elkei+0.25*fedep*fedep*(1+fedep
     *                *(1+0.875*fedep)))
                      lelktmp = lelke
                      IF ((lelec .LT. 0)) THEN
                        dedxmid=ededx1(Lelktmp,MEDIUM)*elktmp+ededx0(Lel
     *                  ktmp,MEDIUM)
                        dedxmid = 1/dedxmid
                        aux = ededx1(lelktmp,medium)*dedxmid
                      ELSE
                        dedxmid=pdedx1(Lelktmp,MEDIUM)*elktmp+pdedx0(Lel
     *                  ktmp,MEDIUM)
                        dedxmid = 1/dedxmid
                        aux = pdedx1(lelktmp,medium)*dedxmid
                      END IF
                      aux = aux*(1+2*aux)*(fedep/(2-fedep))**2/6
                      tuss = fedep*eke*dedxmid*(1+aux)
                      ekei = E_array(lelkef+1,medium)
                      elkei = (lelkef + 1 - eke0(medium))/eke1(medium)
                      fedep = 1 - ekef/ekei
                      elktmp = 0.5*(elkei+elkef+0.25*fedep*fedep*(1+fede
     *                p*(1+0.875*fedep)))
                      lelktmp = lelkef
                      IF ((lelec .LT. 0)) THEN
                        dedxmid=ededx1(Lelktmp,MEDIUM)*elktmp+ededx0(Lel
     *                  ktmp,MEDIUM)
                        dedxmid = 1/dedxmid
                        aux = ededx1(lelktmp,medium)*dedxmid
                      ELSE
                        dedxmid=pdedx1(Lelktmp,MEDIUM)*elktmp+pdedx0(Lel
     *                  ktmp,MEDIUM)
                        dedxmid = 1/dedxmid
                        aux = pdedx1(lelktmp,medium)*dedxmid
                      END IF
                      aux = aux*(1+2*aux)*(fedep/(2-fedep))**2/6
                      tstep = fedep*ekei*dedxmid*(1+aux)
                      tstep=tstep+tuss+ range_ep(qel,lelke,medium)-range
     *                _ep(qel,lelkef+1,medium)
                    END IF
                  END IF
                  total_tstep = tstep
                  compute_tstep = .false.
                END IF
                tstep = total_tstep/rhof
              END IF
              IF ((lelec .LT. 0)) THEN
                dedx0=ededx1(Lelke,MEDIUM)*elke+ededx0(Lelke,MEDIUM)
              ELSE
                dedx0=pdedx1(Lelke,MEDIUM)*elke+pdedx0(Lelke,MEDIUM)
              END IF
              dedx = rhof*dedx0
              tmxs=tmxs1(Lelke,MEDIUM)*elke+tmxs0(Lelke,MEDIUM)
              tmxs = tmxs/rhof
              ekei = E_array(lelke,medium)
              elkei = (lelke - eke0(medium))/eke1(medium)
              fedep = 1 - ekei/eke
              elktmp = 0.5*(elke+elkei+0.25*fedep*fedep*(1+fedep*(1+0.87
     *        5*fedep)))
              lelktmp = lelke
              IF ((lelec .LT. 0)) THEN
                dedxmid=ededx1(Lelktmp,MEDIUM)*elktmp+ededx0(Lelktmp,MED
     *          IUM)
                dedxmid = 1/dedxmid
                aux = ededx1(lelktmp,medium)*dedxmid
              ELSE
                dedxmid=pdedx1(Lelktmp,MEDIUM)*elktmp+pdedx0(Lelktmp,MED
     *          IUM)
                dedxmid = 1/dedxmid
                aux = pdedx1(lelktmp,medium)*dedxmid
              END IF
              aux = aux*(1+2*aux)*(fedep/(2-fedep))**2/6
              range = fedep*eke*dedxmid*(1+aux)
              range = (range + range_ep(qel,lelke,medium))/rhof
              random_tustep = .false.
              IF ((random_tustep)) THEN
                IF (( rng_seed .GT. 24 )) THEN
                  call ranlux(rng_array)
                  rng_seed = 1
                END IF
                rnnotu = rng_array(rng_seed)
                rng_seed = rng_seed + 1
                tmxs = rnnotu*min(tmxs,smaxir(irl))
              ELSE
                tmxs = min(tmxs,smaxir(irl))
              END IF
              tustep = min(tstep,tmxs,range)
              WRITE(6,4380)35
4380          FORMAT(' '/ ' '/ ' ***************************************
     *************************'/ ' *************************************
     ***************************'/ ' '/ ' PRESTA-II is aborting executio
     *n because you have not defined   '/ ' the HOWNEAR macro for your g
     *eometry.                           '/ ' '/ ' You MUST either do so
     * or employ a limited form of PRESTA-II    '/ ' which does not atte
     *mpt the refined boundary crossing or lateral'/ ' correlation featu
     *res of the algorithm.                         '/ ' '/ ' If you inc
     *lude the following macro in your usercode:           '/ ' '/ ' REP
     *LACE {$CALL-HOWNEAR(',a,');} WITH {;}                       '/ ' '
     */ ' you can choose between single scattering mode (very slow) and 
     * '/ ' standard EGS4 mode (no PRESTA enhancments) by the appropriat
     *e  '/ ' choice of the parameters in your input file (see the PREST
     *A-II '/ ' manual)               '/ ' '/ ' ************************
     ****************************************'/ ' **********************
     ******************************************'/ ' '/ ' '/)
              stop
              dnear(np) = tperp
              IF (( i_do_rr(irl) .EQ. 1 .AND. e(np) .LT. e_max_rr(irl) )
     *        ) THEN
                IF ((tperp .GE. range)) THEN
                  idisc = 50 + 49*iq(np)
                  go to 4350
                END IF
              END IF
              blccl = rhof*blcc(medium)
              xccl = rhof*xcc(medium)
              p2 = eke*(eke+rmt2)
              beta2 = p2/(p2 + rmsq)
              IF (( spin_effects )) THEN
                IF ((lelec .LT. 0)) THEN
                  etap=etae_ms1(Lelke,MEDIUM)*elke+etae_ms0(Lelke,MEDIUM
     *            )
                ELSE
                  etap=etap_ms1(Lelke,MEDIUM)*elke+etap_ms0(Lelke,MEDIUM
     *            )
                END IF
                ms_corr=blcce1(Lelke,MEDIUM)*elke+blcce0(Lelke,MEDIUM)
                blccl = blccl/etap/(1+0.25*etap*xccl/blccl/p2)*ms_corr
              END IF
              ssmfp=beta2/blccl
              skindepth = skindepth_for_bca*ssmfp
              tustep = min(tustep,max(tperp,skindepth))
              count_all_steps = count_all_steps + 1
              is_ch_step = .false.
              IF (((tustep .LE. tperp) .AND. ((.NOT.exact_bca) .OR. (tus
     *        tep .GT. skindepth)))) THEN
                callhowfar = .false.
                domultiple = .false.
                dosingle = .false.
                callmsdist = .true.
                tuss = range - range_ep(qel,lelke,medium)/rhof
                IF (( tuss .GE. tustep )) THEN
                  IF (( lelec .LT. 0 )) THEN
                    dedxmid=ededx1(Lelke,MEDIUM)*elke+ededx0(Lelke,MEDIU
     *              M)
                    aux = ededx1(lelke,medium)/dedxmid
                  ELSE
                    dedxmid=pdedx1(Lelke,MEDIUM)*elke+pdedx0(Lelke,MEDIU
     *              M)
                    aux = pdedx1(lelke,medium)/dedxmid
                  END IF
                  de = dedxmid*tustep*rhof
                  fedep = de/eke
                  de = de*(1-0.5*fedep*aux*(1-0.333333*fedep*(aux-1- 0.2
     *            5*fedep*(2-aux*(4-aux)))))
                ELSE
                  lelktmp = lelke
                  tuss = (range - tustep)*rhof
                  IF (( tuss .LE. 0 )) THEN
                    de = eke - TE(medium)*0.99
                  ELSE
4391                IF(tuss.GE.range_ep(qel,lelktmp,medium))GO TO 4392
                      lelktmp = lelktmp - 1
                    GO TO 4391
4392                CONTINUE
                    elktmp = (lelktmp+1-eke0(medium))/eke1(medium)
                    eketmp = E_array(lelktmp+1,medium)
                    tuss = (range_ep(qel,lelktmp+1,medium) - tuss)/rhof
                    IF (( lelec .LT. 0 )) THEN
                      dedxmid=ededx1(Lelktmp,MEDIUM)*elktmp+ededx0(Lelkt
     *                mp,MEDIUM)
                      aux = ededx1(lelktmp,medium)/dedxmid
                    ELSE
                      dedxmid=pdedx1(Lelktmp,MEDIUM)*elktmp+pdedx0(Lelkt
     *                mp,MEDIUM)
                      aux = pdedx1(lelktmp,medium)/dedxmid
                    END IF
                    de = dedxmid*tuss*rhof
                    fedep = de/eketmp
                    de = de*(1-0.5*fedep*aux*(1-0.333333*fedep*(aux-1- 0
     *              .25*fedep*(2-aux*(4-aux)))))
                    de = de + eke - eketmp
                  END IF
                END IF
                tvstep = tustep
                is_ch_step = .true.
                IF ((transport_algorithm .EQ. 0)) THEN
                  call msdist_pII (  eke,de,tustep,rhof,medium,qel,spin_
     *            effects, u(np),v(np),w(np),x(np),y(np),z(np),  uscat,v
     *            scat,wscat,xtrans,ytrans,ztrans,ustep )
                ELSE
                  call msdist_pI (  eke,de,tustep,rhof,medium,qel,spin_e
     *            ffects, u(np),v(np),w(np),x(np),y(np),z(np),  uscat,vs
     *            cat,wscat,xtrans,ytrans,ztrans,ustep )
                END IF
              ELSE
                callmsdist = .false.
                IF ((exact_bca)) THEN
                  domultiple = .false.
                  IF (( rng_seed .GT. 24 )) THEN
                    call ranlux(rng_array)
                    rng_seed = 1
                  END IF
                  rnnoss = rng_array(rng_seed)
                  rng_seed = rng_seed + 1
                  IF (( rnnoss .LT. 1.e-30 )) THEN
                    rnnoss = 1.e-30
                  END IF
                  lambda = - Log(1 - rnnoss)
                  lambda_max = 0.5*blccl*rm/dedx*(eke/rm+1)**3
                  IF (( lambda .GE. 0 .AND. lambda_max .GT. 0 )) THEN
                    IF (( lambda .LT. lambda_max )) THEN
                      tuss=lambda*ssmfp*(1-0.5*lambda/lambda_max)
                    ELSE
                      tuss = 0.5 * lambda * ssmfp
                    END IF
                    IF ((tuss .LT. tustep)) THEN
                      tustep = tuss
                      dosingle = .true.
                    ELSE
                      dosingle = .false.
                    END IF
                  ELSE
                    write(i_log,'(/a)') '***************** Warning: '
                    write(i_log,*) ' lambda > lambda_max: ', lambda,lamb
     *              da_max,' eke dedx: ',eke,dedx, ' ir medium blcc: ',i
     *              r(np),medium,blcc(medium), ' position = ',x(np),y(np
     *              ),z(np)
                    dosingle = .false.
                    np=np-1
                    return
                  END IF
                  ustep = tustep
                ELSE
                  dosingle = .false.
                  domultiple = .true.
                  ekems = eke - 0.5*tustep*dedx
                  p2 = ekems*(ekems+rmt2)
                  beta2 = p2/(p2 + rmsq)
                  chia2 = xccl/(4*blccl*p2)
                  xi = 0.5*xccl/p2/beta2*tustep
                  IF (( spin_effects )) THEN
                    elkems = Log(ekems)
                    Lelkems=eke1(MEDIUM)*elkems+eke0(MEDIUM)
                    IF ((lelec .LT. 0)) THEN
                      etap=etae_ms1(Lelkems,MEDIUM)*elkems+etae_ms0(Lelk
     *                ems,MEDIUM)
                      xi_corr=q1ce_ms1(Lelkems,MEDIUM)*elkems+q1ce_ms0(L
     *                elkems,MEDIUM)
                    ELSE
                      etap=etap_ms1(Lelkems,MEDIUM)*elkems+etap_ms0(Lelk
     *                ems,MEDIUM)
                      xi_corr=q1cp_ms1(Lelkems,MEDIUM)*elkems+q1cp_ms0(L
     *                elkems,MEDIUM)
                    END IF
                    chia2 = chia2*etap
                    xi = xi*xi_corr
                    ms_corr=blcce1(Lelkems,MEDIUM)*elkems+blcce0(Lelkems
     *              ,MEDIUM)
                    blccl = blccl*ms_corr
                  ELSE
                    xi_corr = 1
                    etap = 1
                  END IF
                  xi = xi*(Log(1+1./chia2)-1/(1+chia2))
                  IF (( xi .LT. 0.1 )) THEN
                    ustep = tustep*(1 - xi*(0.5 - xi*0.166667))
                  ELSE
                    ustep = tustep*(1 - Exp(-xi))/xi
                  END IF
                END IF
                IF ((ustep .LT. tperp)) THEN
                  callhowfar = .false.
                ELSE
                  callhowfar = .true.
                END IF
              END IF
            END IF
            irold = ir(np)
            irnew = ir(np)
            idisc = 0
            ustep0 = ustep
            IF ((callhowfar .OR. wt(np) .LE. 0)) THEN
              call howfar
            END IF
            IF ((idisc .GT. 0)) THEN
              go to 4350
            END IF
            IF ((ustep .LE. 0)) THEN
              IF ((ustep .LT. -1e-4)) THEN
                ierust = ierust + 1
                WRITE(6,4400)ierust,ustep,dedx,e(np)-prm, ir(np),irnew,i
     *          rold,x(np),y(np),z(np)
4400            FORMAT(i4,' Negative ustep = ',e12.5,' dedx=',F8.4,' ke=
     *',F8.4, ' ir,irnew,irold =',3i4,' x,y,z =',4e10.3)
                IF ((ierust .GT. 1000)) THEN
                  WRITE(6,4410)
4410              FORMAT(////' Called exit---too many ustep errors'///)
                  call exit(1)
                END IF
              END IF
              ustep = 0
            END IF
            IF ((ustep .EQ. 0 .OR. medium .EQ. 0)) THEN
              IF ((ustep .NE. 0)) THEN
                IF (.false.) THEN
                  edep = pzero
                ELSE
                  vstep = ustep
                  tvstep = vstep
                  edep = pzero
                  e_range = vacdst
                  IARG=0
                  IF ((IAUSFL(IARG+1).NE.0)) THEN
                    CALL AUSGAB(IARG)
                  END IF
                  x(np) = x(np) + u(np)*vstep
                  y(np) = y(np) + v(np)*vstep
                  z(np) = z(np) + w(np)*vstep
                  dnear(np) = dnear(np) - vstep
                END IF
              END IF
              IF ((irnew .NE. irold)) THEN
                ir(np) = irnew
                irl = irnew
                medium = med(irl)
              END IF
              IF ((ustep .NE. 0)) THEN
                IARG=5
                IF ((IAUSFL(IARG+1).NE.0)) THEN
                  CALL AUSGAB(IARG)
                END IF
              END IF
              IF ((eie .LE. ecut(irl))) THEN
                go to 4340
              END IF
              IF ((ustep .NE. 0 .AND. idisc .LT. 0)) THEN
                go to 4350
              END IF
              GO TO 4361
            END IF
            vstep = ustep
            IF ((callhowfar)) THEN
              IF ((exact_bca)) THEN
                tvstep = vstep
                IF ((tvstep .NE. tustep)) THEN
                  dosingle = .false.
                END IF
              ELSE
                IF (( vstep .LT. ustep0 )) THEN
                  ekems = eke - 0.5*tustep*vstep/ustep0*dedx
                  p2 = ekems*(ekems+rmt2)
                  beta2 = p2/(p2 + rmsq)
                  chia2 = xccl/(4*blccl*p2)
                  xi = 0.5*xccl/p2/beta2*vstep
                  IF (( spin_effects )) THEN
                    elkems = Log(ekems)
                    Lelkems=eke1(MEDIUM)*elkems+eke0(MEDIUM)
                    IF ((lelec .LT. 0)) THEN
                      etap=etae_ms1(Lelkems,MEDIUM)*elkems+etae_ms0(Lelk
     *                ems,MEDIUM)
                      xi_corr=q1ce_ms1(Lelkems,MEDIUM)*elkems+q1ce_ms0(L
     *                elkems,MEDIUM)
                    ELSE
                      etap=etap_ms1(Lelkems,MEDIUM)*elkems+etap_ms0(Lelk
     *                ems,MEDIUM)
                      xi_corr=q1cp_ms1(Lelkems,MEDIUM)*elkems+q1cp_ms0(L
     *                elkems,MEDIUM)
                    END IF
                    chia2 = chia2*etap
                    xi = xi*xi_corr
                    ms_corr=blcce1(Lelkems,MEDIUM)*elkems+blcce0(Lelkems
     *              ,MEDIUM)
                    blccl = blccl*ms_corr
                  ELSE
                    xi_corr = 1
                    etap = 1
                  END IF
                  xi = xi*(Log(1+1./chia2)-1/(1+chia2))
                  IF (( xi .LT. 0.1 )) THEN
                    tvstep = vstep*(1 + xi*(0.5 + xi*0.333333))
                  ELSE
                    IF (( xi .LT. 0.999999 )) THEN
                      tvstep = -vstep*Log(1 - xi)/xi
                    ELSE
                      write(i_log,*) ' Stoped in SET-TVSTEP because xi >
     * 1! '
                      write(i_log,*) ' Medium: ',medium
                      write(i_log,*) ' Initial energy: ',eke
                      write(i_log,*) ' Average step energy: ',ekems
                      write(i_log,*) ' tustep: ',tustep
                      write(i_log,*) ' ustep0: ',ustep0
                      write(i_log,*) ' vstep:  ',vstep
                      write(i_log,*) ' ==> xi = ',xi
                      write(i_log,'(/a)') '***************** Error: '
                      write(i_log,*) 'This is a fatal error condition'
                      write(i_log,'(/a)') '***************** Quiting now
     *.'
                      call exit(1)
                    END IF
                  END IF
                ELSE
                  tvstep = tustep
                END IF
              END IF
              tuss = range - range_ep(qel,lelke,medium)/rhof
              IF (( tuss .GE. tvstep )) THEN
                IF (( lelec .LT. 0 )) THEN
                  dedxmid=ededx1(Lelke,MEDIUM)*elke+ededx0(Lelke,MEDIUM)
                  aux = ededx1(lelke,medium)/dedxmid
                ELSE
                  dedxmid=pdedx1(Lelke,MEDIUM)*elke+pdedx0(Lelke,MEDIUM)
                  aux = pdedx1(lelke,medium)/dedxmid
                END IF
                de = dedxmid*tvstep*rhof
                fedep = de/eke
                de = de*(1-0.5*fedep*aux*(1-0.333333*fedep*(aux-1- 0.25*
     *          fedep*(2-aux*(4-aux)))))
              ELSE
                lelktmp = lelke
                tuss = (range - tvstep)*rhof
                IF (( tuss .LE. 0 )) THEN
                  de = eke - TE(medium)*0.99
                ELSE
4421              IF(tuss.GE.range_ep(qel,lelktmp,medium))GO TO 4422
                    lelktmp = lelktmp - 1
                  GO TO 4421
4422              CONTINUE
                  elktmp = (lelktmp+1-eke0(medium))/eke1(medium)
                  eketmp = E_array(lelktmp+1,medium)
                  tuss = (range_ep(qel,lelktmp+1,medium) - tuss)/rhof
                  IF (( lelec .LT. 0 )) THEN
                    dedxmid=ededx1(Lelktmp,MEDIUM)*elktmp+ededx0(Lelktmp
     *              ,MEDIUM)
                    aux = ededx1(lelktmp,medium)/dedxmid
                  ELSE
                    dedxmid=pdedx1(Lelktmp,MEDIUM)*elktmp+pdedx0(Lelktmp
     *              ,MEDIUM)
                    aux = pdedx1(lelktmp,medium)/dedxmid
                  END IF
                  de = dedxmid*tuss*rhof
                  fedep = de/eketmp
                  de = de*(1-0.5*fedep*aux*(1-0.333333*fedep*(aux-1- 0.2
     *            5*fedep*(2-aux*(4-aux)))))
                  de = de + eke - eketmp
                END IF
              END IF
            ELSE
              tvstep = tustep
              IF (( .NOT.callmsdist )) THEN
                tuss = range - range_ep(qel,lelke,medium)/rhof
                IF (( tuss .GE. tvstep )) THEN
                  IF (( lelec .LT. 0 )) THEN
                    dedxmid=ededx1(Lelke,MEDIUM)*elke+ededx0(Lelke,MEDIU
     *              M)
                    aux = ededx1(lelke,medium)/dedxmid
                  ELSE
                    dedxmid=pdedx1(Lelke,MEDIUM)*elke+pdedx0(Lelke,MEDIU
     *              M)
                    aux = pdedx1(lelke,medium)/dedxmid
                  END IF
                  de = dedxmid*tvstep*rhof
                  fedep = de/eke
                  de = de*(1-0.5*fedep*aux*(1-0.333333*fedep*(aux-1- 0.2
     *            5*fedep*(2-aux*(4-aux)))))
                ELSE
                  lelktmp = lelke
                  tuss = (range - tvstep)*rhof
                  IF (( tuss .LE. 0 )) THEN
                    de = eke - TE(medium)*0.99
                  ELSE
4431                IF(tuss.GE.range_ep(qel,lelktmp,medium))GO TO 4432
                      lelktmp = lelktmp - 1
                    GO TO 4431
4432                CONTINUE
                    elktmp = (lelktmp+1-eke0(medium))/eke1(medium)
                    eketmp = E_array(lelktmp+1,medium)
                    tuss = (range_ep(qel,lelktmp+1,medium) - tuss)/rhof
                    IF (( lelec .LT. 0 )) THEN
                      dedxmid=ededx1(Lelktmp,MEDIUM)*elktmp+ededx0(Lelkt
     *                mp,MEDIUM)
                      aux = ededx1(lelktmp,medium)/dedxmid
                    ELSE
                      dedxmid=pdedx1(Lelktmp,MEDIUM)*elktmp+pdedx0(Lelkt
     *                mp,MEDIUM)
                      aux = pdedx1(lelktmp,medium)/dedxmid
                    END IF
                    de = dedxmid*tuss*rhof
                    fedep = de/eketmp
                    de = de*(1-0.5*fedep*aux*(1-0.333333*fedep*(aux-1- 0
     *              .25*fedep*(2-aux*(4-aux)))))
                    de = de + eke - eketmp
                  END IF
                END IF
              END IF
            END IF
            save_de = de
            edep = de
            ekef = eke - de
            eold = eie
            enew = eold - de
            IF (( .NOT.callmsdist )) THEN
              IF (( domultiple )) THEN
                lambda = blccl*tvstep/beta2/etap/(1+chia2)
                xi = xi/xi_corr
                findindex = .true.
                spin_index = .true.
                call mscat(lambda,chia2,xi,elkems,beta2,qel,medium, spin
     *          _effects,findindex,spin_index, costhe,sinthe)
              ELSE
                IF ((dosingle)) THEN
                  ekems = Max(ekef,ecut(irl)-rm)
                  p2 = ekems*(ekems + rmt2)
                  beta2 = p2/(p2 + rmsq)
                  chia2 = xcc(medium)/(4*blcc(medium)*p2)
                  IF (( spin_effects )) THEN
                    elkems = Log(ekems)
                    Lelkems=eke1(MEDIUM)*elkems+eke0(MEDIUM)
                    IF ((lelec .LT. 0)) THEN
                      etap=etae_ms1(Lelkems,MEDIUM)*elkems+etae_ms0(Lelk
     *                ems,MEDIUM)
                    ELSE
                      etap=etap_ms1(Lelkems,MEDIUM)*elkems+etap_ms0(Lelk
     *                ems,MEDIUM)
                    END IF
                    chia2 = chia2*etap
                  END IF
                  call sscat(chia2,elkems,beta2,qel,medium, spin_effects
     *            ,costhe,sinthe)
                ELSE
                  theta = 0
                  sinthe = 0
                  costhe = 1
                END IF
              END IF
            END IF
            e_range = range
            IF (( callmsdist )) THEN
              u_final = uscat
              v_final = vscat
              w_final = wscat
              x_final = xtrans
              y_final = ytrans
              z_final = ztrans
            ELSE
              IF (.NOT.(.false.)) THEN
                x_final = x(np) + u(np)*vstep
                y_final = y(np) + v(np)*vstep
                z_final = z(np) + w(np)*vstep
              END IF
              IF (( domultiple .OR. dosingle )) THEN
                u_tmp = u(np)
                v_tmp = v(np)
                w_tmp = w(np)
                call uphi(2,1)
                u_final = u(np)
                v_final = v(np)
                w_final = w(np)
                u(np) = u_tmp
                v(np) = v_tmp
                w(np) = w_tmp
              ELSE
                u_final = u(np)
                v_final = v(np)
                w_final = w(np)
              END IF
            END IF
            IARG=0
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
            x(np) = x_final
            y(np) = y_final
            z(np) = z_final
            u(np) = u_final
            v(np) = v_final
            w(np) = w_final
            dnear(np) = dnear(np) - vstep
            irold = ir(np)
            peie = peie - edep
            eie = peie
            e(np) = peie
            IF (( irnew .EQ. irl .AND. eie .LE. ecut(irl))) THEN
              go to 4340
            END IF
            medold = medium
            IF ((medium .NE. 0)) THEN
              ekeold = eke
              eke = eie - rm
              elke = log(eke)
              Lelke=eke1(MEDIUM)*elke+eke0(MEDIUM)
            END IF
            IF ((irnew .NE. irold)) THEN
              ir(np) = irnew
              irl = irnew
              medium = med(irl)
            END IF
            IARG=5
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
            IF ((eie .LE. ecut(irl))) THEN
              go to 4340
            END IF
            IF ((idisc .LT. 0)) THEN
              go to 4350
            END IF
            IF((medium .NE. medold))GO TO 4361
            demfp = demfp - save_de*sig
            total_de = total_de - save_de
            total_tstep = total_tstep - tvstep*rhof
            IF (( total_tstep .LT. 1e-9 )) THEN
              demfp = 0
            END IF
            IF(((demfp .LT. 1.E-8)))GO TO4372
          GO TO 4371
4372      CONTINUE
          IF ((lelec .LT. 0)) THEN
            sigf=esig1(Lelke,MEDIUM)*elke+esig0(Lelke,MEDIUM)
            dedx0=ededx1(Lelke,MEDIUM)*elke+ededx0(Lelke,MEDIUM)
            sigf = sigf/dedx0
          ELSE
            sigf=psig1(Lelke,MEDIUM)*elke+psig0(Lelke,MEDIUM)
            dedx0=pdedx1(Lelke,MEDIUM)*elke+pdedx0(Lelke,MEDIUM)
            sigf = sigf/dedx0
          END IF
          sigratio = sigf/sig0
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          rfict = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          IF(((rfict .LE. sigratio)))GO TO4362
        GO TO 4361
4362    CONTINUE
        IF ((lelec .LT. 0)) THEN
          ebr1=ebr11(Lelke,MEDIUM)*elke+ebr10(Lelke,MEDIUM)
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          rnno24 = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          IF ((rnno24 .LE. ebr1)) THEN
            go to 4440
          ELSE
            IF ((e(np) .LE. thmoll(medium) .AND. eii_flag .EQ. 0)) THEN
              IF ((ebr1 .LE. 0)) THEN
                go to 4330
              END IF
              go to 4440
            END IF
            IARG=8
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
            call moller
            IARG=9
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
            IF((iq(np) .EQ. 0))return
          END IF
          go to 4330
        END IF
        pbr1=pbr11(Lelke,MEDIUM)*elke+pbr10(Lelke,MEDIUM)
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        rnno25 = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        IF ((rnno25 .LT. pbr1)) THEN
          go to 4440
        END IF
        pbr2=pbr21(Lelke,MEDIUM)*elke+pbr20(Lelke,MEDIUM)
        IF ((rnno25 .LT. pbr2)) THEN
          IARG=10
          IF ((IAUSFL(IARG+1).NE.0)) THEN
            CALL AUSGAB(IARG)
          END IF
          call bhabha
          IARG=11
          IF ((IAUSFL(IARG+1).NE.0)) THEN
            CALL AUSGAB(IARG)
          END IF
          IF((iq(np) .EQ. 0))return
        ELSE
          IARG=12
          IF ((IAUSFL(IARG+1).NE.0)) THEN
            CALL AUSGAB(IARG)
          END IF
          call annih
          IARG=13
          IF ((IAUSFL(IARG+1).NE.0)) THEN
            CALL AUSGAB(IARG)
          END IF
          GO TO 4332
        END IF
      GO TO 4331
4332  CONTINUE
      return
4440  IARG=6
      IF ((IAUSFL(IARG+1).NE.0)) THEN
        CALL AUSGAB(IARG)
      END IF
      call brems
      IARG=7
      IF ((IAUSFL(IARG+1).NE.0)) THEN
        CALL AUSGAB(IARG)
      END IF
      IF ((iq(np) .EQ. 0)) THEN
        return
      ELSE
        go to 4330
      END IF
4340  IF (( medium .GT. 0 )) THEN
        IF ((eie .GT. ae(medium))) THEN
          idr = 1
          IF ((lelec .LT. 0)) THEN
            edep = e(np) - prm
          ELSE
            EDEP=PEIE-PRM
          END IF
        ELSE
          idr = 2
          edep = e(np) - prm
        END IF
      ELSE
        idr = 1
        edep = e(np) - prm
      END IF
      IARG=idr
      IF ((IAUSFL(IARG+1).NE.0)) THEN
        CALL AUSGAB(IARG)
      END IF
4450  CONTINUE
      IF ((lelec .GT. 0)) THEN
        IF ((edep .LT. peie)) THEN
          IARG=28
          IF ((IAUSFL(IARG+1).NE.0)) THEN
            CALL AUSGAB(IARG)
          END IF
          call annih_at_rest
          IARG=14
          IF ((IAUSFL(IARG+1).NE.0)) THEN
            CALL AUSGAB(IARG)
          END IF
          return
        END IF
      END IF
      np = np - 1
      ircode = 2
      return
4350  idisc = abs(idisc)
      IF (((lelec .LT. 0) .OR. (idisc .EQ. 99))) THEN
        edep = e(np) - prm
      ELSE
        edep = e(np) + prm
      END IF
      IARG=3
      IF ((IAUSFL(IARG+1).NE.0)) THEN
        CALL AUSGAB(IARG)
      END IF
      IF((idisc .EQ. 99))goto 4450
      np = np - 1
      ircode = 2
      return
      end
      SUBROUTINE HATCH
      implicit none
      character*512 toUpper
      COMMON/QDEBUG/QDEBUG
      LOGICAL QDEBUG
      COMMON/BOUNDS/ECUT(1),PCUT(1),VACDST
      real*8 ECUT,  PCUT,  VACDST
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      COMMON/ELECIN/ esig_e(1),psig_e(1), esige_max, psige_max, range_ep
     *(0:1,500,1), E_array(500,1), etae_ms0(500,1),etae_ms1(500,1),etap_
     *ms0(500,1),etap_ms1(500,1),q1ce_ms0(500,1),q1ce_ms1(500,1),q1cp_ms
     *0(500,1),q1cp_ms1(500,1),q2ce_ms0(500,1),q2ce_ms1(500,1),q2cp_ms0(
     *500,1),q2cp_ms1(500,1),blcce0(500,1),blcce1(500,1), EKE0(1),EKE1(1
     *), XR0(1),TEFF0(1),BLCC(1),XCC(1), ESIG0(500,1),ESIG1(500,1),PSIG0
     *(500,1),PSIG1(500,1),EDEDX0(500,1),EDEDX1(500,1),PDEDX0(500,1),PDE
     *DX1(500,1),EBR10(500,1),EBR11(500,1),PBR10(500,1),PBR11(500,1),PBR
     *20(500,1),PBR21(500,1),TMXS0(500,1),TMXS1(500,1), expeke1(1), IUNR
     *ST(1),EPSTFL(1),IAPRIM(1), sig_ismonotone(0:1,1)
      real*8 esig_e,   psig_e,   esige_max,  psige_max,  range_ep,  E_ar
     *ray,  etae_ms0,etae_ms1,  etap_ms0,etap_ms1,  q1ce_ms0,q1ce_ms1,
     *q1cp_ms0,q1cp_ms1,  q2ce_ms0,q2ce_ms1,  q2cp_ms0,q2cp_ms1,  blcce0
     *,blcce1,   expeke1,  EKE0,EKE1, XR0,  TEFF0,  BLCC,  XCC,  ESIG0,E
     *SIG1,  PSIG0,PSIG1,  EDEDX0,EDEDX1,  PDEDX0,PDEDX1,  EBR10,EBR11,
     * PBR10,PBR11,  PBR20,PBR21,  TMXS0,TMXS1
      integer*4 IUNRST,  EPSTFL,  IAPRIM
      logical sig_ismonotone
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/MISC/  DUNIT,KMPI,KMPO,RHOR(1),MED(1),IRAYLR(1),IPHOTONUCR(
     *1)
      real*8 DUNIT,  RHOR
      integer*4 KMPI,  KMPO
      integer*2 MED,  IRAYLR,  IPHOTONUCR
      COMMON/PHOTIN/ EBINDA(1), GE0(1),GE1(1), GMFP0(2000,1),GMFP1(2000,
     *1),GBR10(2000,1),GBR11(2000,1),GBR20(2000,1),GBR21(2000,1), RCO0(1
     *),RCO1(1), RSCT0(100,1),RSCT1(100,1), COHE0(2000,1),COHE1(2000,1),
     *  PHOTONUC0(2000,1),PHOTONUC1(2000,1), DPMFP, MPGEM(1,1), NGR(1)
      real*8 EBINDA,  GE0,GE1,  GMFP0,GMFP1,  GBR10,GBR11,  GBR20,GBR21,
     *  RCO0,RCO1,  RSCT0,RSCT1,  COHE0,COHE1,   PHOTONUC0,PHOTONUC1,  D
     *PMFP
      integer*4
     *                  MPGEM,
     *                          NGR
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/UPHIIN/SINC0,SINC1,SIN0(1002),SIN1(1002)
      real*8 SINC0,SINC1,SIN0,SIN1
      COMMON/UPHIOT/THETA,SINTHE,COSTHE,SINPHI, COSPHI,PI,TWOPI,PI5D2
      real*8 THETA,  SINTHE,  COSTHE,  SINPHI,  COSPHI,  PI,TWOPI,PI5D2
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      common/ET_control/ smaxir(1),estepe,ximax,  skindepth_for_bca,tran
     *sport_algorithm, bca_algorithm,exact_bca,spin_effects
      real*8 smaxir,  estepe,  ximax,      skindepth_for_bca
      integer*4 transport_algorithm, bca_algorithm
      logical exact_bca,  spin_effects
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      common/x_options/eadl_relax,  mcdf_pe_xsections
      logical eadl_relax, mcdf_pe_xsections
      common/egs_vr/ e_max_rr(1),  prob_RR,  nbr_split,  i_play_RR,
     * i_survived_RR,
     *     n_RR_warning,                                        i_do_rr(
     *1)
      real*8          e_max_rr,prob_RR
      integer*4       nbr_split,i_play_RR,i_survived_RR,n_RR_warning
      integer*2     i_do_rr
      CHARACTER*4 MBUF(72),MDLABL(8)
      real*8 ACD ,  ADEV ,  ASD ,  COST ,  CTHET ,  DEL ,  DFACT ,  DFAC
     *TI,  DUNITO,  DUNITR,  FNSSS ,  P ,  PZNORM,  RDEV ,  S2C2 ,  S2C2
     *MN,  S2C2MX,  SINT ,  SX ,  SXX ,  SXY ,   SY ,   WID ,  XS ,  XS0
     * ,  XS1 ,  XSI ,  WSS ,  YS ,  ZEROS(3)
      integer*4 I ,  I1ST ,  IB ,  ID ,  IE ,  IL ,  IM ,  IRAYL ,  IRN
     *,  ISTEST,  ISUB ,  ISS ,  IZ ,   IZZ ,  J ,  JR ,  LCTHET,  LMDL
     *,  LMDN ,  LTHETA,  MD ,  MXSINC,  NCMFP ,   NEKE ,   NGE ,   NGRI
     *M ,  NISUB ,  NLEKE ,    NM ,  NRANGE,    NRNA ,  NSEKE ,   NSGE ,
     *   NSINSS,  LOK(1)
      character*256 tmp_string
      integer*4 lnblnk1
      DATA MDLABL/' ','M','E','D','I','U','M','='/,LMDL/8/,LMDN/24/,DUNI
     *TO/1./
      DATA I1ST/1/,NSINSS/37/,MXSINC/1002/,ISTEST/0/,NRNA/1000/
4460  FORMAT(1X,14I5)
4470  FORMAT(1X,1PE14.5,4E14.5)
4480  FORMAT(72A1)
      IF ((I1ST.NE.0)) THEN
        I1ST=0
        DO 4491 J=1,1
          IF ((SMAXIR(J).LE.0.0)) THEN
            SMAXIR(J)=1E10
          END IF
4491    CONTINUE
4492    CONTINUE
        NISUB=MXSINC-2
        FNSSS=NSINSS
        WID=PI5D2/FLOAT(NISUB)
        WSS=WID/(FNSSS-1.0)
        ZEROS(1)=0.
        ZEROS(2)=PI
        ZEROS(3)=TWOPI
        DO 4501 ISUB=1,MXSINC
          SX=0.
          SY=0.
          SXX=0.
          SXY=0.
          XS0=WID*FLOAT(ISUB-2)
          XS1=XS0+WID
          IZ=0
          DO 4511 IZZ=1,3
            IF (((XS0.LE.ZEROS(IZZ)).AND.(ZEROS(IZZ).LE.XS1))) THEN
              IZ=IZZ
              GO TO4512
            END IF
4511      CONTINUE
4512      CONTINUE
          IF ((IZ.EQ.0)) THEN
            XSI=XS0
          ELSE
            XSI=ZEROS(IZ)
          END IF
          DO 4521 ISS=1,NSINSS
            XS=WID*FLOAT(ISUB-2)+WSS*FLOAT(ISS-1)-XSI
            YS=SIN(XS+XSI)
            SX=SX+XS
            SY=SY+YS
            SXX=SXX+XS*XS
            SXY=SXY+XS*YS
4521      CONTINUE
4522      CONTINUE
          IF ((IZ.NE.0)) THEN
            SIN1(ISUB)=SXY/SXX
            SIN0(ISUB)=-SIN1(ISUB)*XSI
          ELSE
            DEL=FNSSS*SXX-SX*SX
            SIN1(ISUB)=(FNSSS*SXY-SY*SX)/DEL
            SIN0(ISUB)=(SY*SXX-SX*SXY)/DEL - SIN1(ISUB)*XSI
          END IF
4501    CONTINUE
4502    CONTINUE
        SINC0=2.0
        SINC1=1.0/WID
        IF ((ISTEST.NE.0)) THEN
          ADEV=0.
          RDEV=0.
          S2C2MN=10.
          S2C2MX=0.
          DO 4531 ISUB=1,NISUB
            DO 4541 ISS=1,NSINSS
              THETA=WID*FLOAT(ISUB-1)+WSS*FLOAT(ISS-1)
              CTHET=PI5D2-THETA
              SINTHE=sin(THETA)
              COSTHE=sin(CTHET)
              SINT=SIN(THETA)
              COST=COS(THETA)
              ASD=ABS(SINTHE-SINT)
              ACD=ABS(COSTHE-COST)
              ADEV=max(ADEV,ASD,ACD)
              IF((SINT.NE.0.0))RDEV=max(RDEV,ASD/ABS(SINT))
              IF((COST.NE.0.0))RDEV=max(RDEV,ACD/ABS(COST))
              S2C2=SINTHE**2+COSTHE**2
              S2C2MN=min(S2C2MN,S2C2)
              S2C2MX=max(S2C2MX,S2C2)
              IF ((ISUB.LT.11)) THEN
                write(i_log,'(1PE20.7,4E20.7)') THETA,SINTHE,SINT,COSTHE
     *          ,COST
              END IF
4541        CONTINUE
4542        CONTINUE
4531      CONTINUE
4532      CONTINUE
          write(i_log,'(a,2i5)') ' SINE TESTS,MXSINC,NSINSS=',MXSINC,NSI
     *    NSS
          write(i_log,'(a,1PE16.8,3e16.8)') ' ADEV,RDEV,S2C2(MN,MX) =',
     *    ADEV,RDEV,S2C2MN,S2C2MX
          ADEV=0.
          RDEV=0.
          S2C2MN=10.
          S2C2MX=0.
          DO 4551 IRN=1,NRNA
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            THETA = rng_array(rng_seed)
            rng_seed = rng_seed + 1
            THETA=THETA*PI5D2
            CTHET=PI5D2-THETA
            SINTHE=sin(THETA)
            COSTHE=sin(CTHET)
            SINT=SIN(THETA)
            COST=COS(THETA)
            ASD=ABS(SINTHE-SINT)
            ACD=ABS(COSTHE-COST)
            ADEV=max(ADEV,ASD,ACD)
            IF((SINT.NE.0.0))RDEV=max(RDEV,ASD/ABS(SINT))
            IF((COST.NE.0.0))RDEV=max(RDEV,ACD/ABS(COST))
            S2C2=SINTHE**2+COSTHE**2
            S2C2MN=min(S2C2MN,S2C2)
            S2C2MX=max(S2C2MX,S2C2)
4551      CONTINUE
4552      CONTINUE
          write(i_log,'(a,i7,a)') ' TEST AT ',NRNA,' RANDOM ANGLES IN (0
     *,5*PI/2)'
          write(i_log,'(1PE16.8,3E16.8)') ' ADEV,RDEV,S2C2(MN,MX) =', AD
     *    EV,RDEV,S2C2MN,S2C2MX
        END IF
        P=1.
        DO 4561 I=1,50
          PWR2I(I)=P
          P=P/2.
4561    CONTINUE
4562    CONTINUE
      END IF
      DO 4571 J=1,NMED
4580    CONTINUE
          DO 4581 I=1,1
          IF ((IRAYLR(I).EQ.1.AND.MED(I).EQ.J)) THEN
            IRAYLM(J)=1
            GO TO 4582
          END IF
4581    CONTINUE
4582    CONTINUE
4571  CONTINUE
4572  CONTINUE
      IPHOTONUC=0
      DO 4591 J=1,NMED
4600    CONTINUE
          DO 4601 I=1,1
          IF ((IPHOTONUCR(I).EQ.1.AND.MED(I).EQ.J)) THEN
            IPHOTONUCM(J)=1
            IPHOTONUC=1
            GO TO 4602
          END IF
4601    CONTINUE
4602    CONTINUE
4591  CONTINUE
4592  CONTINUE
      write(i_log,'(a,i3)') ' ===> Photonuclear flag: ', iphotonuc
      IF((.NOT.is_pegsless))REWIND KMPI
      NM=0
      DO 4611 IM=1,NMED
        LOK(IM)=0
        IF ((IRAYLM(IM).EQ.1)) THEN
          write(i_log,'(a,i3/)') ' RAYLEIGH OPTION REQUESTED FOR MEDIUM
     *NUMBER',IM
        END IF
4611  CONTINUE
4612  CONTINUE
      DO 4621 IM=1,NMED
        IF ((IPHOTONUCM(IM).EQ.1)) THEN
          write(i_log,'(a,i3/)') ' PHOTONUCLEAR REQUESTED FOR MEDIUM NUM
     *BER',IM
        END IF
4621  CONTINUE
4622  CONTINUE
      IF ((.NOT.is_pegsless)) THEN
4630    CONTINUE
4631      CONTINUE
4640      CONTINUE
4641        CONTINUE
            READ(KMPI,4480,END=4650)MBUF
            DO 4661 IB=1,LMDL
              IF((MBUF(IB).NE.MDLABL(IB)))GO TO 4641
4661        CONTINUE
4662        CONTINUE
4670        CONTINUE
              DO 4671 IM=1,NMED
              DO 4681 IB=1,LMDN
                IL=LMDL+IB
                IF((MBUF(IL).NE.MEDIA(IB,IM)))GO TO 4671
                IF((IB.EQ.LMDN))GO TO 4642
4681          CONTINUE
4682          CONTINUE
4671        CONTINUE
4672        CONTINUE
          GO TO 4641
4642      CONTINUE
          IF((LOK(IM).NE.0))GO TO 4640
          LOK(IM)=1
          NM=NM+1
          read(kmpi,'(a)',err=4690) tmp_string
          goto 4700
4690      write(i_log,'(/a)') '***************** Error: '
          write(i_log,*) 'Error while reading pegs4 file'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
4700      CONTINUE
          read(tmp_string,1,ERR=4710)  (MBUF(I),I=1,5),RHO(IM),NNE(IM),I
     *    UNRST(IM),EPSTFL(IM),IAPRIM(IM)
1         FORMAT(5A1,5X,F11.0,4X,I2,9X,I1,9X,I1,9X,I1)
          GO TO 4720
4710      CONTINUE
          write(i_log,*) 'Found medium with gas pressure'
          read(tmp_string,2) (MBUF(I),I=1,5),RHO(IM),NNE(IM),IUNRST(IM),
     *    EPSTFL(IM), IAPRIM(IM)
2         FORMAT(5A1,5X,F11.0,4X,I2,26X,I1,9X,I1,9X,I1)
4720      CONTINUE
            DO 4721 IE=1,NNE(IM)
            READ(KMPI,4730)(MBUF(I),I=1,6),(ASYM(IM,IE,I),I=1,2), ZELEM(
     *      IM,IE),WA(IM,IE),PZ(IM,IE),RHOZ(IM,IE)
4730        FORMAT (6A1,2A1,3X,F3.0,3X,F9.0,4X,F12.0,6X,F12.0)
4721      CONTINUE
4722      CONTINUE
          READ(KMPI,4470) RLC(IM),AE(IM),AP(IM),UE(IM),UP(IM)
          TE(IM)=AE(IM)-RM
          THMOLL(IM)=TE(IM)*2. + RM
          READ(KMPI,4460) MSGE(IM),MGE(IM),MSEKE(IM),MEKE(IM),MLEKE(IM),
     *    MCMFP(IM),MRANGE(IM),IRAYL
          NSGE=MSGE(IM)
          NGE=MGE(IM)
          NSEKE=MSEKE(IM)
          NEKE=MEKE(IM)
          NLEKE=MLEKE(IM)
          NCMFP=MCMFP(IM)
          NRANGE=MRANGE(IM)
          READ(KMPI,4470)(DL1(I,IM),DL2(I,IM),DL3(I,IM),DL4(I,IM),DL5(I,
     *    IM),DL6(I,IM),I=1,6)
          READ(KMPI,4470)DELCM(IM),(ALPHI(I,IM),BPAR(I,IM),DELPOS(I,IM),
     *    I=1,2)
          READ(KMPI,4470)XR0(IM),TEFF0(IM),BLCC(IM),XCC(IM)
          READ(KMPI,4470)EKE0(IM),EKE1(IM)
          READ(KMPI,4470) (ESIG0(I,IM),ESIG1(I,IM),PSIG0(I,IM),PSIG1(I,I
     *    M),EDEDX0(I,IM),EDEDX1(I,IM),PDEDX0(I,IM),PDEDX1(I,IM),EBR10(I
     *    ,IM),EBR11(I,IM),PBR10(I,IM),PBR11(I,IM),PBR20(I,IM),PBR21(I,I
     *    M),TMXS0(I,IM),TMXS1(I,IM),I=1,NEKE)
          READ(KMPI,4470)EBINDA(IM),GE0(IM),GE1(IM)
          READ(KMPI,4470)(GMFP0(I,IM),GMFP1(I,IM),GBR10(I,IM),GBR11(I,IM
     *    ),GBR20(I,IM),GBR21(I,IM),I=1,NGE)
          IF ((IRAYL.EQ.1)) THEN
            READ(KMPI,4460) NGR(IM)
            NGRIM=NGR(IM)
            READ(KMPI,4470)RCO0(IM),RCO1(IM)
            READ(KMPI,4470)(RSCT0(I,IM),RSCT1(I,IM),I=1,NGRIM)
            READ(KMPI,4470)(COHE0(I,IM),COHE1(I,IM),I=1,NGE)
            write(i_log,'(a,i3,a)') ' Rayleigh data available for medium
     *', IM, ' in PEGS4 data set.'
          END IF
          IF ((IRAYLM(IM).EQ.1)) THEN
            IF ((IRAYL.NE.1)) THEN
              IF ((toUpper(photon_xsections(:lnblnk1(photon_xsections)))
     *        .EQ.'PEGS4')) THEN
                write(i_log,'(/a)') '***************** Error: '
                write(i_log,'(a,i3 /,a /,a)') ' IN HATCH: REQUESTED RAYL
     *EIGH OPTION FOR MEDIUM', IM,' BUT RAYLEIGH DATA NOT INCLUDED IN PE
     *GS4 FILE.', ' YOU WILL NOT BE ABLE TO USE THE PEGS4 DATA WITH RAYL
     *EIGH ON!'
                write(i_log,'(/a)') '***************** Quiting now.'
                call exit(1)
              ELSE
                write(i_log,'(/a)') '***************** Warning: '
                write(i_log,'(a,i3 /,a)') ' IN HATCH: REQUESTED RAYLEIGH
     * OPTION FOR MEDIUM', IM,' BUT RAYLEIGH DATA NOT INCLUDED IN PEGS4
     *FILE.'
              END IF
            ELSE
              IF ((toUpper(photon_xsections(:lnblnk1(photon_xsections)))
     *        .EQ.'PEGS4')) THEN
                call egs_init_rayleigh_sampling(IM)
              END IF
            END IF
          END IF
          IF((NM.GE.NMED))GO TO4632
        GO TO 4631
4632    CONTINUE
        CLOSE (UNIT=KMPI)
        DUNITR=DUNIT
        IF ((DUNIT.LT.0.0)) THEN
          ID=MAX0(1,MIN0(1,int(-DUNIT)))
          DUNIT=RLC(ID)
        END IF
        IF ((DUNIT.NE.1.0)) THEN
          write(i_log,'(a,1PE14.5,E14.5,a)') ' DUNIT REQUESTED&USED ARE:
     * ', DUNITR,DUNIT,'(CM.)'
        END IF
        DO 4741 IM=1,NMED
          DFACT=RLC(IM)/DUNIT
          DFACTI=1.0/DFACT
          I=1
            GO TO 4753
4751        I=I+1
4753        IF(I-(MEKE(IM)).GT.0)GO TO 4752
            ESIG0(I,IM)=ESIG0(I,IM)*DFACTI
            ESIG1(I,IM)=ESIG1(I,IM)*DFACTI
            PSIG0(I,IM)=PSIG0(I,IM)*DFACTI
            PSIG1(I,IM)=PSIG1(I,IM)*DFACTI
            EDEDX0(I,IM)=EDEDX0(I,IM)*DFACTI
            EDEDX1(I,IM)=EDEDX1(I,IM)*DFACTI
            PDEDX0(I,IM)=PDEDX0(I,IM)*DFACTI
            PDEDX1(I,IM)=PDEDX1(I,IM)*DFACTI
            TMXS0(I,IM)=TMXS0(I,IM)*DFACT
            TMXS1(I,IM)=TMXS1(I,IM)*DFACT
          GO TO 4751
4752      CONTINUE
          TEFF0(IM)=TEFF0(IM)*DFACT
          BLCC(IM)=BLCC(IM)*DFACTI
          XCC(IM)=XCC(IM)*SQRT(DFACTI)
          RLDU(IM)=RLC(IM)/DUNIT
          I=1
            GO TO 4763
4761        I=I+1
4763        IF(I-(MGE(IM)).GT.0)GO TO 4762
            GMFP0(I,IM)=GMFP0(I,IM)*DFACT
            GMFP1(I,IM)=GMFP1(I,IM)*DFACT
          GO TO 4761
4762      CONTINUE
4741    CONTINUE
4742    CONTINUE
        VACDST=VACDST*DUNITO/DUNIT
        DUNITO=DUNIT
      ELSE
        write(i_log,*) ' PEGSLESS INPUT.  CALCULATING ELECTRON CROSS-SEC
     *TIONS.'
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,'(a/a)') ' Code cannot be run in pegsless mode.', '
     *Compile with required files and try again.'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      DO 4771 JR=1,1
        MD=MED(JR)
        IF (((MD.GE.1).AND.(MD.LE.NMED))) THEN
          ECUT(JR)=max(ECUT(JR),AE(MD))
          PCUT(JR)=max(PCUT(JR),AP(MD))
          IF ((RHOR(JR).EQ.0.0)) THEN
            RHOR(JR)=RHO(MD)
          END IF
        END IF
4771  CONTINUE
4772  CONTINUE
      IF ((IBRDST.EQ.1)) THEN
        DO 4781 IM=1,NMED
          ZBRANG(IM)=0.0
          PZNORM=0.0
          DO 4791 IE=1,NNE(IM)
            ZBRANG(IM)= ZBRANG(IM)+PZ(IM,IE)*ZELEM(IM,IE)*(ZELEM(IM,IE)+
     *      1.0)
            PZNORM=PZNORM+PZ(IM,IE)
4791      CONTINUE
4792      CONTINUE
          ZBRANG(IM)=(8.116224E-05)*(ZBRANG(IM)/PZNORM)**(1./3.)
          LZBRANG(IM)=-log(ZBRANG(IM))
4781    CONTINUE
4782    CONTINUE
      END IF
      IF ((IPRDST.GT.0)) THEN
        DO 4801 IM=1,NMED
          ZBRANG(IM)=0.0
          PZNORM=0.0
          DO 4811 IE=1,NNE(IM)
            ZBRANG(IM)= ZBRANG(IM)+PZ(IM,IE)*ZELEM(IM,IE)*(ZELEM(IM,IE)+
     *      1.0)
            PZNORM=PZNORM+PZ(IM,IE)
4811      CONTINUE
4812      CONTINUE
          ZBRANG(IM)=(8.116224E-05)*(ZBRANG(IM)/PZNORM)**(1./3.)
4801    CONTINUE
4802    CONTINUE
      END IF
      IF ((toUpper(photon_xsections(:lnblnk1(photon_xsections))) .EQ. 'P
     *EGS4')) THEN
        write(i_log,'(/a)') '***************** Warning: '
        write(i_log,'(6(a/))') 'Using photon data from PEGS4 file!!!', '
     *However, the new Rayleigh angular sampling will be used.', 'The or
     *iginal EGS4 angular sampling undersamples large scattering ', 'ang
     *les. This may have little impact as Rayleigh scattering ', 'is for
     *ward peaked.', '**************************************************
     ********'
      ELSE
        call egs_init_user_photon(photon_xsections,comp_xsections, photo
     *  nuc_xsections,xsec_out)
      END IF
      call mscati
      IF (( eadl_relax .AND. photon_xsections .EQ. 'xcom' )) THEN
        call init_compton
        call EDGSET(1,1)
      ELSE
        call EDGSET(1,1)
        call init_compton
      END IF
      IF (( xsec_out .EQ. 1 .AND. eadl_relax)) THEN
        call egs_print_binding_energies
      END IF
      call fix_brems
      IF (( ibr_nist .GE. 1 )) THEN
        call init_nist_brems
      END IF
      IF (( pair_nrc .EQ. 1 )) THEN
        call init_nrc_pair
      END IF
      call eii_init
      call init_triplet
      IF ((NMED.EQ.1)) THEN
        write(i_log,*) 'EGSnrc SUCCESSFULLY ''HATCHED'' FOR ONE MEDIUM.'
      ELSE
        write(i_log,'(a,i5,a)') 'EGSnrc SUCCESSFULLY ''HATCHED'' FOR ',N
     *  MED,' MEDIA.'
      END IF
      RETURN
4650  write(i_log,'(a,i2//,a/,a/)') ' END OF FILE ON UNIT ',KMPI, ' PROG
     *RAM STOPPED IN HATCH BECAUSE THE', ' FOLLOWING NAMES WERE NOT RECO
     *GNIZED:'
      DO 4821 IM=1,NMED
        IF ((LOK(IM).NE.1)) THEN
          write(i_log,'(40x,a,24a1,a)') '''',(MEDIA(I,IM),I=1,LMDN),''''
        END IF
4821  CONTINUE
4822  CONTINUE
      STOP
      END
      subroutine fix_brems
      implicit none
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      common/nist_brems/ nb_fdata(0:50,100,1), nb_xdata(0:50,100,1), nb_
     *wdata(50,100,1), nb_idata(50,100,1), nb_emin(1),nb_emax(1), nb_lem
     *in(1),nb_lemax(1), nb_dle(1),nb_dlei(1), log_ap(1)
      real*8 nb_fdata,nb_xdata,nb_wdata,nb_emin,nb_emax,nb_lemin,nb_lema
     *x, nb_dle,nb_dlei,log_ap
      integer*4 nb_idata
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      integer*4 medium,i
      real*8 Zt,Zb,Zf,Zg,Zv,fmax1,fmax2,Zi,pi,fc,xi,aux, XSIF,FCOULC
      DO 4831 medium=1,nmed
        log_ap(medium) = log(ap(medium))
        Zt = 0
        Zb = 0
        Zf = 0
        DO 4841 i=1,NNE(medium)
          Zi = ZELEM(medium,i)
          pi = PZ(medium,i)
          fc = FCOULC(Zi)
          xi = XSIF(Zi)
          aux = pi*Zi*(Zi + xi)
          Zt = Zt + aux
          Zb = Zb - aux*Log(Zi)/3
          Zf = Zf + aux*fc
4841    CONTINUE
4842    CONTINUE
        Zv = (Zb - Zf)/Zt
        Zg = Zb/Zt
        fmax1 = 2*(20.863 + 4*Zg) - 2*(20.029 + 4*Zg)/3
        fmax2 = 2*(20.863 + 4*Zv) - 2*(20.029 + 4*Zv)/3
        dl1(1,medium) = (20.863 + 4*Zg)/fmax1
        dl2(1,medium) = -3.242/fmax1
        dl3(1,medium) = 0.625/fmax1
        dl4(1,medium) = (21.12+4*Zg)/fmax1
        dl5(1,medium) = -4.184/fmax1
        dl6(1,medium) = 0.952
        dl1(2,medium) = (20.029+4*Zg)/fmax1
        dl2(2,medium) = -1.93/fmax1
        dl3(2,medium) = -0.086/fmax1
        dl4(2,medium) = (21.12+4*Zg)/fmax1
        dl5(2,medium) = -4.184/fmax1
        dl6(2,medium) = 0.952
        dl1(3,medium) = (20.863 + 4*Zv)/fmax2
        dl2(3,medium) = -3.242/fmax2
        dl3(3,medium) = 0.625/fmax2
        dl4(3,medium) = (21.12+4*Zv)/fmax2
        dl5(3,medium) = -4.184/fmax2
        dl6(3,medium) = 0.952
        dl1(4,medium) = (20.029+4*Zv)/fmax2
        dl2(4,medium) = -1.93/fmax2
        dl3(4,medium) = -0.086/fmax2
        dl4(4,medium) = (21.12+4*Zv)/fmax2
        dl5(4,medium) = -4.184/fmax2
        dl6(4,medium) = 0.952
        dl1(5,medium) = (3*(20.863 + 4*Zg) - (20.029 + 4*Zg))
        dl2(5,medium) = (3*(-3.242) - (-1.930))
        dl3(5,medium) = (3*(0.625)-(-0.086))
        dl4(5,medium) = (2*21.12+8*Zg)
        dl5(5,medium) = (2*(-4.184))
        dl6(5,medium) = 0.952
        dl1(6,medium) = (3*(20.863 + 4*Zg) + (20.029 + 4*Zg))
        dl2(6,medium) = (3*(-3.242) + (-1.930))
        dl3(6,medium) = (3*0.625+(-0.086))
        dl4(6,medium) = (4*21.12+16*Zg)
        dl5(6,medium) = (4*(-4.184))
        dl6(6,medium) = 0.952
        dl1(7,medium) = (3*(20.863 + 4*Zv) - (20.029 + 4*Zv))
        dl2(7,medium) = (3*(-3.242) - (-1.930))
        dl3(7,medium) = (3*(0.625)-(-0.086))
        dl4(7,medium) = (2*21.12+8*Zv)
        dl5(7,medium) = (2*(-4.184))
        dl6(7,medium) = 0.952
        dl1(8,medium) = (3*(20.863 + 4*Zv) + (20.029 + 4*Zv))
        dl2(8,medium) = (3*(-3.242) + (-1.930))
        dl3(8,medium) = (3*0.625+(-0.086))
        dl4(8,medium) = (4*21.12+16*Zv)
        dl5(8,medium) = (4*(-4.184))
        dl6(8,medium) = 0.952
        bpar(2,medium) = dl1(7,medium)/(3*dl1(8,medium) + dl1(7,medium))
        bpar(1,medium) = 12*dl1(8,medium)/(3*dl1(8,medium) + dl1(7,mediu
     *  m))
4831  CONTINUE
4832  CONTINUE
      return
      end
      real*8 function FCOULC(Z)
      implicit none
      real*8 Z
      real*8 fine,asq
      data fine/137.03604/
      asq = Z/fine
      asq = asq*asq
      FCOULC = asq*(1.0/(1.0+ASQ)+0.20206+ASQ*(-0.0369+ASQ*(0.0083+ASQ*(
     *-0.002))))
      return
      end
      real*8 function XSIF(Z)
      implicit none
      real*8 Z
      integer*4 iZ
      real*8 alrad(4),alradp(4),a1440,a183,FCOULC
      data alrad/5.31,4.79,4.74,4.71/
      data alradp/6.144,5.621,5.805,5.924/
      data a1440/1194.0/,A183/184.15/
      IF (( Z .LE. 4 )) THEN
        iZ = Z
        xsif = alradp(iZ)/(alrad(iZ) - FCOULC(Z))
      ELSE
        xsif = Log(A1440*Z**(-0.666667))/(Log(A183*Z**(-0.33333))-FCOULC
     *  (Z))
      END IF
      return
      end
      subroutine init_compton
      implicit none
      common/compton_data/ iz_array(1538),  be_array(1538),  Jo_array(15
     *38),  erfJo_array(1538),   ne_array(1538),  shn_array(1538),
     *shell_array(200,1), eno_array(200,1), eno_atbin_array(200,1), n_sh
     *ell(1), radc_flag,  ibcmp(1)
      integer*4 iz_array,ne_array,shn_array,eno_atbin_array, shell_array
     *,n_shell,radc_flag
      real*8 be_array,Jo_array,erfJo_array,eno_array
      integer*2 ibcmp
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      COMMON/EDGE/binding_energies(30,100), interaction_prob(6,100), rel
     *axation_prob(39,100), edge_energies(16,100), edge_number(100), edg
     *e_a(16,100), edge_b(16,100), edge_c(16,100), edge_d(16,100), IEDGF
     *L(1),IPHTER(1)
      real*8 binding_energies,  interaction_prob,    relaxation_prob,  e
     *dge_energies,  edge_a,edge_b,edge_c,edge_d
      integer*2 IEDGFL,  IPHTER
      integer*4 edge_number
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/MISC/  DUNIT,KMPI,KMPO,RHOR(1),MED(1),IRAYLR(1),IPHOTONUCR(
     *1)
      real*8 DUNIT,  RHOR
      integer*4 KMPI,  KMPO
      integer*2 MED,  IRAYLR,  IPHOTONUCR
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      common/x_options/eadl_relax,  mcdf_pe_xsections
      logical eadl_relax, mcdf_pe_xsections
      integer*4 i,j,iz,nsh,j_l,j_h
      real*8 aux,pztot,atav
      real*8 aux_erf,erf1
      logical getd
      IF (( radc_flag .EQ. 1 )) THEN
        write(i_log,'(/a)') '***************** Warning: '
        write(i_log,*) 'You are trying to use radiative Compton correcti
     *ons'
        write(i_log,*) 'without having included rad_compton1.mortran'
        write(i_log,'(a//)') 'Turning radiative Compton corrections OFF
     *...'
        radc_flag = 0
      END IF
      getd = .false.
      DO 4851 j=1,1
        medium = med(j)
        IF (( medium .GT. 0 .AND. medium .LE. nmed)) THEN
          IF (( ibcmp(j) .GT. 0 )) THEN
            getd = .true.
            GO TO4852
          END IF
        END IF
4851  CONTINUE
4852  CONTINUE
      IF (( .NOT.getd )) THEN
        IF (( eadl_relax .AND. photon_xsections .EQ. 'xcom' )) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,'(a,/a,/a)') 'You must turn ON Compton binding cor
     *rections when using', 'a detailed atomic relaxation (eadl_relax=tr
     *ue) since ', 'binding energies taken from incoh.data below 1 keV!'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        write(i_log,'(a/)') ' Bound Compton scattering not requested! '
        return
      END IF
      write(i_log,'(/a$)') 'Bound Compton scattering requested, reading
     *data ......'
      rewind(i_incoh)
      DO 4861 j=1,18
        read(i_incoh,*)
4861  CONTINUE
4862  CONTINUE
      iz = 0
      DO 4871 j=1,1538
        read(i_incoh,*) iz_array(j),shn_array(j),ne_array(j), Jo_array(j
     *  ),be_array(j)
        Jo_array(j) = Jo_array(j)*137.
        be_array(j) = be_array(j)*1e-6/PRM
        aux_erf = 0.70710678119*(1+0.3*Jo_array(j))
        erfJo_array(j) = 0.82436063535*(erf1(aux_erf)-1)
        IF ((eadl_relax)) THEN
          IF ((iz_array(j) .NE. iz)) THEN
            shn_array(j) = 1
            iz = iz_array(j)
          ELSE
            shn_array(j) = shn_array(j-1)+1
          END IF
          IF ((binding_energies(shn_array(j),iz_array(j)) .GT. 0)) THEN
            be_array(j) = binding_energies(shn_array(j),iz_array(j))/PRM
          ELSE IF((photon_xsections .EQ. 'xcom')) THEN
            binding_energies(shn_array(j),iz_array(j)) = be_array(j)*PRM
          END IF
        END IF
4871  CONTINUE
4872  CONTINUE
      write(i_log,*) ' Done'
      write(i_log,'(/a)') ' Initializing Bound Compton scattering ......
     *'
      DO 4881 medium=1,nmed
        pztot = 0
        nsh = 0
        DO 4891 i=1,nne(medium)
          iz = int(zelem(medium,i))
          DO 4901 j=1,1538
            IF (( iz .EQ. iz_array(j) )) THEN
              nsh = nsh + 1
              IF (( nsh .GT. 200 )) THEN
                write(i_log,'(/a)') '***************** Error: '
                write(i_log,'(/a,i3,a,i4,a/,a)') ' For medium ',medium,
     *          ' the number of shells is > ',200,'!', ' Increase the pa
     *rameter $MXMDSH! '
                write(i_log,'(/a)') '***************** Quiting now.'
                call exit(1)
              END IF
              shell_array(nsh,medium) = j
              aux = pz(medium,i)*ne_array(j)
              eno_array(nsh,medium) = aux
              pztot = pztot + aux
            END IF
4901      CONTINUE
4902      CONTINUE
4891    CONTINUE
4892    CONTINUE
        IF (( nsh .EQ. 0 )) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,'(a,i3,a)') ' Medium ',medium,' has zero shells! '
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        n_shell(medium) = nsh
        write(i_log,'(a,i3,a,i3,a)') ' Medium ',medium,' has ',nsh,' she
     *lls: '
        DO 4911 i=1,nsh
          j = shell_array(i,medium)
          eno_array(i,medium) = eno_array(i,medium)/pztot
          write(i_log,'(i4,i5,i4,f9.5,e10.3,f10.3)') i,j,shn_array(j),en
     *    o_array(i,medium), Jo_array(j),be_array(j)*PRM*1000.
          eno_array(i,medium) = -eno_array(i,medium)
          eno_atbin_array(i,medium) = i
4911    CONTINUE
4912    CONTINUE
        atav = 1./nsh
        DO 4921 i=1,nsh-1
          DO 4931 j_h=1,nsh-1
            IF (( eno_array(j_h,medium) .LT. 0 )) THEN
              IF((abs(eno_array(j_h,medium)) .GT. atav))GO TO4932
            END IF
4931      CONTINUE
4932      CONTINUE
          DO 4941 j_l=1,nsh-1
            IF (( eno_array(j_l,medium) .LT. 0 )) THEN
              IF((abs(eno_array(j_l,medium)) .LT. atav))GO TO4942
            END IF
4941      CONTINUE
4942      CONTINUE
          aux = atav - abs(eno_array(j_l,medium))
          eno_array(j_h,medium) = eno_array(j_h,medium) + aux
          eno_array(j_l,medium) = -eno_array(j_l,medium)/atav + j_l
          eno_atbin_array(j_l,medium) = j_h
          IF((i .EQ. nsh-1))eno_array(j_h,medium) = 1 + j_h
4921    CONTINUE
4922    CONTINUE
        DO 4951 i=1,nsh
          IF (( eno_array(i,medium) .LT. 0 )) THEN
            eno_array(i,medium) = 1 + i
          END IF
4951    CONTINUE
4952    CONTINUE
4881  CONTINUE
4882  CONTINUE
      write(i_log,'(a/)') ' ...... Done.'
      getd = .false.
      DO 4961 j=1,1
        IF (( iedgfl(j) .GT. 0 .AND. iedgfl(j) .LE. 100 )) THEN
          getd = .true.
          GO TO4962
        END IF
4961  CONTINUE
4962  CONTINUE
      IF((getd))return
      write(i_log,'(/a)') '***************** Error: '
      write(i_log,'(/a,/a,/a,/a)') ' In subroutine init_compton: ', '
     *Scattering off bound electrons creates atomic vacancies,', '   pot
     *entially starting an atomic relaxation cascade. ', '   Please turn
     * ON atomic relaxations.'
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
      return
      end
      SUBROUTINE MOLLER
      implicit none
      COMMON/QDEBUG/QDEBUG
      LOGICAL QDEBUG
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/UPHIOT/THETA,SINTHE,COSTHE,SINPHI, COSPHI,PI,TWOPI,PI5D2
      real*8 THETA,  SINTHE,  COSTHE,  SINPHI,  COSPHI,  PI,TWOPI,PI5D2
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      common/egs_vr/ e_max_rr(1),  prob_RR,  nbr_split,  i_play_RR,
     * i_survived_RR,
     *     n_RR_warning,                                        i_do_rr(
     *1)
      real*8          e_max_rr,prob_RR
      integer*4       nbr_split,i_play_RR,i_survived_RR,n_RR_warning
      integer*2     i_do_rr
      common/eii_data/ eii_xsection_a( 10000),  eii_xsection_b( 10000),
     * eii_cons(1), eii_a(40),  eii_b(40),  eii_L_factor,  eii_z(40),  e
     *ii_sh(40),  eii_nshells(100),  eii_nsh(1),  eii_first(1,50),  eii_
     *no(1,50),  eii_flag
      real*8 eii_xsection_a,eii_xsection_b,eii_a,eii_b,eii_cons,eii_L_fa
     *ctor
      integer*4 eii_z,eii_sh,eii_nshells
      integer*4 eii_first,eii_no
      integer*4 eii_elements,eii_flag,eii_nsh
      COMMON/ELECIN/ esig_e(1),psig_e(1), esige_max, psige_max, range_ep
     *(0:1,500,1), E_array(500,1), etae_ms0(500,1),etae_ms1(500,1),etap_
     *ms0(500,1),etap_ms1(500,1),q1ce_ms0(500,1),q1ce_ms1(500,1),q1cp_ms
     *0(500,1),q1cp_ms1(500,1),q2ce_ms0(500,1),q2ce_ms1(500,1),q2cp_ms0(
     *500,1),q2cp_ms1(500,1),blcce0(500,1),blcce1(500,1), EKE0(1),EKE1(1
     *), XR0(1),TEFF0(1),BLCC(1),XCC(1), ESIG0(500,1),ESIG1(500,1),PSIG0
     *(500,1),PSIG1(500,1),EDEDX0(500,1),EDEDX1(500,1),PDEDX0(500,1),PDE
     *DX1(500,1),EBR10(500,1),EBR11(500,1),PBR10(500,1),PBR11(500,1),PBR
     *20(500,1),PBR21(500,1),TMXS0(500,1),TMXS1(500,1), expeke1(1), IUNR
     *ST(1),EPSTFL(1),IAPRIM(1), sig_ismonotone(0:1,1)
      real*8 esig_e,   psig_e,   esige_max,  psige_max,  range_ep,  E_ar
     *ray,  etae_ms0,etae_ms1,  etap_ms0,etap_ms1,  q1ce_ms0,q1ce_ms1,
     *q1cp_ms0,q1cp_ms1,  q2ce_ms0,q2ce_ms1,  q2cp_ms0,q2cp_ms1,  blcce0
     *,blcce1,   expeke1,  EKE0,EKE1, XR0,  TEFF0,  BLCC,  XCC,  ESIG0,E
     *SIG1,  PSIG0,PSIG1,  EDEDX0,EDEDX1,  PDEDX0,PDEDX1,  EBR10,EBR11,
     * PBR10,PBR11,  PBR20,PBR21,  TMXS0,TMXS1
      integer*4 IUNRST,  EPSTFL,  IAPRIM
      logical sig_ismonotone
      COMMON/EPCONT/EDEP,EDEP_LOCAL,TSTEP,TUSTEP,USTEP,TVSTEP,VSTEP, RHO
     *F,EOLD,ENEW,EKE,ELKE,GLE,E_RANGE, x_final,y_final,z_final, u_final
     *,v_final,w_final, IDISC,IROLD,IRNEW,IAUSFL(35)
      DOUBLE PRECISION EDEP,  EDEP_LOCAL
      real*8 TSTEP,  TUSTEP,  USTEP,  VSTEP,  TVSTEP,  RHOF,  EOLD,  ENE
     *W,  EKE,  ELKE,  GLE,  E_RANGE, x_final,y_final,z_final,  u_final,
     *v_final,w_final
      integer*4 IDISC,  IROLD,  IRNEW,  IAUSFL
      COMMON/EDGE/binding_energies(30,100), interaction_prob(6,100), rel
     *axation_prob(39,100), edge_energies(16,100), edge_number(100), edg
     *e_a(16,100), edge_b(16,100), edge_c(16,100), edge_d(16,100), IEDGF
     *L(1),IPHTER(1)
      real*8 binding_energies,  interaction_prob,    relaxation_prob,  e
     *dge_energies,  edge_a,edge_b,edge_c,edge_d
      integer*2 IEDGFL,  IPHTER
      integer*4 edge_number
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      DOUBLE PRECISION PEIE,  PEKSE2,  PESE1,  PESE2,  PEKIN,  H1,  DCOS
     *TH
      real*8 EIE,  EKIN,  T0,  E0,  EXTRAE,  E02,  EP0,  G2,G3,  GMAX,
     *BR,  R,  REJF4,  RNNO27,  RNNO28,  ESE1,  ESE2
      real*8 sigm,pbrem,rsh,Uj,sig_j
      integer*4 lelke,iele,ish,nsh,ifirst,i,jj,iZ,iarg
      NPold = NP
      PEIE=E(NP)
      EIE=PEIE
      PEKIN=PEIE-PRM
      EKIN=PEKIN
      IF (( eii_flag .GT. 0 .AND. eii_nsh(medium) .GT. 0 )) THEN
        Lelke=eke1(MEDIUM)*elke+eke0(MEDIUM)
        sigm=esig1(Lelke,MEDIUM)*elke+esig0(Lelke,MEDIUM)
        pbrem=ebr11(Lelke,MEDIUM)*elke+ebr10(Lelke,MEDIUM)
        sigm = sigm*(1 - pbrem)
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        rsh = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        rsh = sigm*rsh
        DO 4971 iele=1,nne(medium)
          iZ = int(zelem(medium,iele)+0.5)
          nsh = eii_no(medium,iele)
          IF (( nsh .GT. 0 )) THEN
            ifirst = eii_first(medium,iele)
            DO 4981 ish=1,nsh
              Uj = binding_energies(ish,iZ)
              IF (( ekin .GT. Uj .AND. (Uj .GT. te(medium) .OR. Uj .GT.
     *        ap(medium)) )) THEN
                jj = ifirst + ish - 1
                i = eii_a(jj)*elke + eii_b(jj) + (jj-1)*250
                sig_j = eii_xsection_a(i)*elke + eii_xsection_b(i)
                sig_j = sig_j*pz(medium,iele)*eii_cons(medium)
                rsh = rsh - sig_j
                IF (( rsh .LT. 0 )) THEN
                  IARG=31
                  IF ((IAUSFL(IARG+1).NE.0)) THEN
                    CALL AUSGAB(IARG)
                  END IF
                  call eii_sample(ish,iZ,Uj)
                  IARG=32
                  IF ((IAUSFL(IARG+1).NE.0)) THEN
                    CALL AUSGAB(IARG)
                  END IF
                  return
                END IF
              END IF
4981        CONTINUE
4982        CONTINUE
          END IF
4971    CONTINUE
4972    CONTINUE
      END IF
      IF((ekin .LE. 2*te(medium)))return
      T0=EKIN/RM
      E0=T0+1.0
      EXTRAE = EIE - THMOLL(MEDIUM)
      E02=E0*E0
      EP0=TE(MEDIUM)/EKIN
      G2=T0*T0/E02
      G3=(2.*T0+1.)/E02
      GMAX=(1.+1.25*G2)
4991  CONTINUE
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        RNNO27 = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        BR = TE(MEDIUM)/(EKIN-EXTRAE*RNNO27)
        R=BR/(1.-BR)
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        RNNO28 = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        REJF4=(1.+G2*BR*BR+R*(R-G3))
        RNNO28=GMAX*RNNO28
        IF((RNNO28.LE.REJF4))GO TO4992
      GO TO 4991
4992  CONTINUE
      PEKSE2=BR*EKIN
      PESE1=PEIE-PEKSE2
      PESE2=PEKSE2+PRM
      ESE1=PESE1
      ESE2=PESE2
      E(NP)=PESE1
      IF (( np+1 .GT. 2 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,'(//,3a,/,2(a,i9))') ' In subroutine ','MOLLER', ' s
     *tack size exceeded! ',' $MAXSTACK = ',2,' np = ',np+1
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      E(NP+1)=PESE2
      H1=(PEIE+PRM)/PEKIN
      DCOSTH=H1*(PESE1-PRM)/(PESE1+PRM)
      SINTHE=DSQRT(1.D0-DCOSTH)
      COSTHE=DSQRT(DCOSTH)
      CALL UPHI(2,1)
      NP=NP+1
      IQ(NP)=-1
      DCOSTH=H1*(PESE2-PRM)/(PESE2+PRM)
      SINTHE=-DSQRT(1.D0-DCOSTH)
      COSTHE=DSQRT(DCOSTH)
      CALL UPHI(3,2)
      RETURN
      END
      subroutine mscati
      implicit none
      real*8 ededx,ei,eil,eip1,eip1l,si,sip1,eke,elke,aux,ecutmn,tstbm,t
     *stbmn
      real*8 p2,beta2,dedx0,ekef,elkef,estepx,ektmp,elktmp,chi_a2
      integer*4 i,leil,leip1l,neke,lelke,lelkef,lelktmp
      logical ise_monoton, isp_monoton
      real*8 sigee,sigep,sig,sige_old,sigp_old
      COMMON/BOUNDS/ECUT(1),PCUT(1),VACDST
      real*8 ECUT,  PCUT,  VACDST
      COMMON/ELECIN/ esig_e(1),psig_e(1), esige_max, psige_max, range_ep
     *(0:1,500,1), E_array(500,1), etae_ms0(500,1),etae_ms1(500,1),etap_
     *ms0(500,1),etap_ms1(500,1),q1ce_ms0(500,1),q1ce_ms1(500,1),q1cp_ms
     *0(500,1),q1cp_ms1(500,1),q2ce_ms0(500,1),q2ce_ms1(500,1),q2cp_ms0(
     *500,1),q2cp_ms1(500,1),blcce0(500,1),blcce1(500,1), EKE0(1),EKE1(1
     *), XR0(1),TEFF0(1),BLCC(1),XCC(1), ESIG0(500,1),ESIG1(500,1),PSIG0
     *(500,1),PSIG1(500,1),EDEDX0(500,1),EDEDX1(500,1),PDEDX0(500,1),PDE
     *DX1(500,1),EBR10(500,1),EBR11(500,1),PBR10(500,1),PBR11(500,1),PBR
     *20(500,1),PBR21(500,1),TMXS0(500,1),TMXS1(500,1), expeke1(1), IUNR
     *ST(1),EPSTFL(1),IAPRIM(1), sig_ismonotone(0:1,1)
      real*8 esig_e,   psig_e,   esige_max,  psige_max,  range_ep,  E_ar
     *ray,  etae_ms0,etae_ms1,  etap_ms0,etap_ms1,  q1ce_ms0,q1ce_ms1,
     *q1cp_ms0,q1cp_ms1,  q2ce_ms0,q2ce_ms1,  q2cp_ms0,q2cp_ms1,  blcce0
     *,blcce1,   expeke1,  EKE0,EKE1, XR0,  TEFF0,  BLCC,  XCC,  ESIG0,E
     *SIG1,  PSIG0,PSIG1,  EDEDX0,EDEDX1,  PDEDX0,PDEDX1,  EBR10,EBR11,
     * PBR10,PBR11,  PBR20,PBR21,  TMXS0,TMXS1
      integer*4 IUNRST,  EPSTFL,  IAPRIM
      logical sig_ismonotone
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/MISC/  DUNIT,KMPI,KMPO,RHOR(1),MED(1),IRAYLR(1),IPHOTONUCR(
     *1)
      real*8 DUNIT,  RHOR
      integer*4 KMPI,  KMPO
      integer*2 MED,  IRAYLR,  IPHOTONUCR
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      common/ET_control/ smaxir(1),estepe,ximax,  skindepth_for_bca,tran
     *sport_algorithm, bca_algorithm,exact_bca,spin_effects
      real*8 smaxir,  estepe,  ximax,      skindepth_for_bca
      integer*4 transport_algorithm, bca_algorithm
      logical exact_bca,  spin_effects
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      IF (( bca_algorithm .EQ. 0 )) THEN
        exact_bca = .true.
      ELSE
        exact_bca = .false.
      END IF
      IF (( estepe .LE. 0 .OR. estepe .GE. 1)) THEN
        estepe = 0.25
      END IF
      IF (( ximax .LE. 0 .OR. ximax .GE. 1 )) THEN
        IF (( exact_bca )) THEN
          ximax = 0.5
        ELSE
          ximax = 0.5
        END IF
      END IF
      IF ((transport_algorithm .NE. 0 .AND. transport_algorithm .NE. 1 .
     *AND. transport_algorithm .NE. 2 )) THEN
        transport_algorithm = 0
      END IF
      IF (( skindepth_for_bca .LE. 1e-4 )) THEN
        IF (( .NOT.exact_bca )) THEN
          write(i_log,*) ' old PRESTA calculates default min. step-size
     *for BCA: '
          ecutmn = 1e30
          DO 5001 i=1,1
            IF (( med(i) .GT. 0 .AND. med(i) .LE. nmed )) THEN
              ecutmn = Min(ecutmn,ecut(i))
            END IF
5001      CONTINUE
5002      CONTINUE
          write(i_log,*) '     minimum ECUT found: ',ecutmn
          tstbmn = 1e30
          DO 5011 medium=1,nmed
            tstbm = (ecutmn-prm)*(ecutmn+prm)/ecutmn**2
            tstbm = blcc(medium)*tstbm*(ecutmn/xcc(medium))**2
            aux = Log(tstbm)
            IF((aux .GT. 300))write(i_log,*) 'aux > 300 ? ',aux
            tstbm = Log(tstbm/aux)
            tstbmn = Min(tstbmn,tstbm)
5011      CONTINUE
5012      CONTINUE
          write(i_log,*) '     default BLCMIN is: ',tstbmn
          skindepth_for_bca = Exp(tstbmn)
          write(i_log,*) '     this corresponds to ',skindepth_for_bca,
     *    ' elastic MFPs '
        ELSE
          skindepth_for_bca = 3
        END IF
      END IF
      call init_ms_SR
      DO 5021 medium=1,nmed
        blcc(medium) = 1.16699413758864573*blcc(medium)
        xcc(medium) = xcc(medium)**2
5021  CONTINUE
5022  CONTINUE
      IF (( spin_effects )) THEN
        call init_spin
      END IF
      write(i_log,*) ' '
      esige_max = 0
      psige_max = 0
      DO 5031 medium=1,nmed
        sigee = 1E-15
        sigep = 1E-15
        neke = meke(medium)
        ise_monoton = .true.
        isp_monoton = .true.
        sige_old = -1
        sigp_old = -1
        DO 5041 i=1,neke
          ei = exp((float(i) - eke0(medium))/eke1(medium))
          eil = log(ei)
          leil = i
          ededx=ededx1(Leil,MEDIUM)*eil+ededx0(Leil,MEDIUM)
          sig=esig1(Leil,MEDIUM)*eil+esig0(Leil,MEDIUM)
          sig = sig/ededx
          IF((sig .GT. sigee))sigee = sig
          IF((sig .LT. sige_old))ise_monoton = .false.
          sige_old = sig
          ededx=pdedx1(Leil,MEDIUM)*eil+pdedx0(Leil,MEDIUM)
          sig=psig1(Leil,MEDIUM)*eil+psig0(Leil,MEDIUM)
          sig = sig/ededx
          IF((sig .GT. sigep))sigep = sig
          IF((sig .LT. sigp_old))isp_monoton = .false.
          sigp_old = sig
5041    CONTINUE
5042    CONTINUE
        write(i_log,*) ' Medium ',medium,' sige = ',sigee,sigep,' monoto
     *ne = ', ise_monoton,isp_monoton
        sig_ismonotone(0,medium) = ise_monoton
        sig_ismonotone(1,medium) = isp_monoton
        esig_e(medium) = sigee
        psig_e(medium) = sigep
        IF((sigee .GT. esige_max))esige_max = sigee
        IF((sigep .GT. psige_max))psige_max = sigep
5031  CONTINUE
5032  CONTINUE
      write(i_log,*) ' '
      write(i_log,*) ' Initializing tmxs for estepe = ',estepe,' and xim
     *ax = ',ximax
      write(i_log,*) ' '
      DO 5051 medium=1,nmed
        ei = exp((1 - eke0(medium))/eke1(medium))
        eil = log(ei)
        leil = 1
        E_array(1,medium) = ei
        expeke1(medium) = Exp(1./eke1(medium))-1
        range_ep(0,1,medium) = 0
        range_ep(1,1,medium) = 0
        neke = meke(medium)
        DO 5061 i=1,neke - 1
          eip1 = exp((float(i + 1) - eke0(medium))/eke1(medium))
          E_array(i+1,medium) = eip1
          eke = 0.5*(eip1+ei)
          elke = Log(eke)
          Lelke=eke1(MEDIUM)*elke+eke0(MEDIUM)
          ededx=pdedx1(Lelke,MEDIUM)*elke+pdedx0(Lelke,MEDIUM)
          aux = pdedx1(i,medium)/ededx
          range_ep(1,i+1,medium) = range_ep(1,i,medium) + (eip1-ei)/eded
     *    x*(1+aux*(1+2*aux)*((eip1-ei)/eke)**2/24)
          ededx=ededx1(Lelke,MEDIUM)*elke+ededx0(Lelke,MEDIUM)
          aux = ededx1(i,medium)/ededx
          range_ep(0,i+1,medium) = range_ep(0,i,medium) + (eip1-ei)/eded
     *    x*(1+aux*(1+2*aux)*((eip1-ei)/eke)**2/24)
          ei = eip1
5061    CONTINUE
5062    CONTINUE
        eil = (1 - eke0(medium))/eke1(medium)
        ei = Exp(eil)
        leil = 1
        p2 = ei*(ei+2*rm)
        beta2 = p2/(p2+rm*rm)
        chi_a2 = Xcc(medium)/(4*p2*blcc(medium))
        dedx0=ededx1(Leil,MEDIUM)*eil+ededx0(Leil,MEDIUM)
        estepx = 2*p2*beta2*dedx0/ei/Xcc(medium)/(Log(1+1./chi_a2)*(1+ch
     *  i_a2)-1)
        estepx = estepx*ximax
        IF (( estepx .GT. estepe )) THEN
          estepx = estepe
        END IF
        si = estepx*ei/dedx0
        DO 5071 i=1,neke - 1
          elke = (i + 1 - eke0(medium))/eke1(medium)
          eke = Exp(elke)
          lelke = i+1
          p2 = eke*(eke+2*rm)
          beta2 = p2/(p2+rm*rm)
          chi_a2 = Xcc(medium)/(4*p2*blcc(medium))
          ededx=ededx1(Lelke,MEDIUM)*elke+ededx0(Lelke,MEDIUM)
          estepx = 2*p2*beta2*ededx/eke/ Xcc(medium)/(Log(1+1./chi_a2)*(
     *    1+chi_a2)-1)
          estepx = estepx*ximax
          IF (( estepx .GT. estepe )) THEN
            estepx = estepe
          END IF
          ekef = (1-estepx)*eke
          IF (( ekef .LE. E_array(1,medium) )) THEN
            sip1 = (E_array(1,medium) - ekef)/dedx0
            ekef = E_array(1,medium)
            elkef = (1 - eke0(medium))/eke1(medium)
            lelkef = 1
          ELSE
            elkef = Log(ekef)
            Lelkef=eke1(MEDIUM)*elkef+eke0(MEDIUM)
            leip1l = lelkef + 1
            eip1l = (leip1l - eke0(medium))/eke1(medium)
            eip1 = E_array(leip1l,medium)
            aux = (eip1 - ekef)/eip1
            elktmp = 0.5*(elkef+eip1l+0.25*aux*aux*(1+aux*(1+0.875*aux))
     *      )
            ektmp = 0.5*(ekef+eip1)
            lelktmp = lelkef
            ededx=ededx1(Lelktmp,MEDIUM)*elktmp+ededx0(Lelktmp,MEDIUM)
            aux = ededx1(lelktmp,medium)/ededx
            sip1 = (eip1 - ekef)/ededx*( 1+aux*(1+2*aux)*((eip1-ekef)/ek
     *      tmp)**2/24)
          END IF
          sip1 = sip1 + range_ep(0,lelke,medium) - range_ep(0,lelkef+1,m
     *    edium)
          tmxs1(i,medium) = (sip1 - si)*eke1(medium)
          tmxs0(i,medium) = sip1 - tmxs1(i,medium)*elke
          si = sip1
5071    CONTINUE
5072    CONTINUE
        tmxs0(neke,medium) = tmxs0(neke - 1,medium)
        tmxs1(neke,medium) = tmxs1(neke - 1,medium)
5051  CONTINUE
5052  CONTINUE
      return
      end
      subroutine mscat(lambda,chia2,q1,elke,beta2,qel,medium, spin_effec
     *ts,find_index,spin_index, cost,sint)
      implicit none
      real*8 lambda, chia2,q1,elke,beta2,cost,sint
      integer*4 qel,medium
      logical spin_effects,find_index,spin_index
      common/ms_data/ ums_array(0:63,0:7,0:31), fms_array(0:63,0:7,0:31)
     *, wms_array(0:63,0:7,0:31), ims_array(0:63,0:7,0:31), llammin,llam
     *max,dllamb,dllambi,dqms,dqmsi
      real*4 ums_array,fms_array,wms_array, llammin,llammax,dllamb,dllam
     *bi,dqms,dqmsi
      integer*2 ims_array
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      real*8 sprob,explambda,wsum,wprob,xi,rejf,spin_rejection, cosz,sin
     *z,phi,omega2,llmbda,ai,aj,ak,a,u,du,x1,rnno
      integer*4 icount,i,j,k
      save i,j,omega2
      IF ((lambda .LE. 13.8)) THEN
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        sprob = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        explambda = Exp(-lambda)
        IF ((sprob .LT. explambda)) THEN
          cost = 1
          sint = 0
          return
        END IF
        wsum = (1+lambda)*explambda
        IF (( sprob .LT. wsum )) THEN
5080      CONTINUE
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          xi = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          xi = 2*chia2*xi/(1 - xi + chia2)
          cost = 1 - xi
          IF (( spin_effects )) THEN
            rejf = spin_rejection(qel,medium,elke,beta2,q1,cost, spin_in
     *      dex,.false.)
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            rnno = rng_array(rng_seed)
            rng_seed = rng_seed + 1
            IF (( rnno .GT. rejf )) THEN
              GOTO 5080
            END IF
          END IF
          sint = sqrt(xi*(2 - xi))
          return
        END IF
        IF (( lambda .LE. 1 )) THEN
          wprob = explambda
          wsum = explambda
          cost = 1
          sint = 0
          icount = 0
5091      CONTINUE
            icount = icount + 1
            IF((icount .GT. 20))GO TO5092
            wprob = wprob*lambda/icount
            wsum = wsum + wprob
5100        CONTINUE
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            xi = rng_array(rng_seed)
            rng_seed = rng_seed + 1
            xi = 2*chia2*xi/(1 - xi + chia2)
            cosz = 1 - xi
            IF (( spin_effects )) THEN
              rejf = spin_rejection(qel,medium,elke,beta2,q1,cosz, spin_
     *        index,.false.)
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              rnno = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              IF (( rnno .GT. rejf )) THEN
                GOTO 5100
              END IF
            END IF
            sinz = xi*(2 - xi)
            IF (( sinz .GT. 1.e-20 )) THEN
              sinz = Sqrt(sinz)
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              xi = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              phi = xi*6.2831853
              cost = cost*cosz - sint*sinz*Cos(phi)
              sint = Sqrt(Max(0.0,(1-cost)*(1+cost)))
            END IF
            IF((( wsum .GT. sprob)))GO TO5092
          GO TO 5091
5092      CONTINUE
          return
        END IF
      END IF
      IF ((lambda .LE. 1e5 )) THEN
        IF ((find_index)) THEN
          llmbda = log(lambda)
          ai = llmbda*dllambi
          i = ai
          ai = ai - i
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          xi = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          IF((xi .LT. ai))i = i + 1
          IF (( q1 .LT. 1e-3 )) THEN
            j = 0
          ELSE IF(( q1 .LT. 0.5 )) THEN
            aj = q1*dqmsi
            j = aj
            aj = aj - j
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            xi = rng_array(rng_seed)
            rng_seed = rng_seed + 1
            IF((xi .LT. aj))j = j + 1
          ELSE
            j = 7
          END IF
          IF ((llmbda .LT. 2.2299)) THEN
            omega2 = chia2*(lambda + 4)*(1.347006 + llmbda*( 0.209364 -
     *      llmbda*(0.45525 - llmbda*(0.50142 - 0.081234*llmbda))))
          ELSE
            omega2 = chia2*(lambda + 4)*(-2.77164 + llmbda*(2.94874 - ll
     *      mbda*(0.1535754 - llmbda*0.00552888)))
          END IF
          find_index = .false.
        END IF
5110    CONTINUE
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        xi = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        ak = xi*31
        k = ak
        ak = ak - k
        IF((ak .GT. wms_array(i,j,k)))k = ims_array(i,j,k)
        a = fms_array(i,j,k)
        u = ums_array(i,j,k)
        du = ums_array(i,j,k+1) - u
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        xi = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        IF (( abs(a) .LT. 0.2 )) THEN
          x1 = 0.5*(1-xi)*a
          u = u + xi*du*(1+x1*(1-xi*a))
        ELSE
          u = u - du/a*(1-Sqrt(1+xi*a*(2+a)))
        END IF
        xi = omega2*u/(1 + 0.5*omega2 - u)
        IF (( xi .GT. 1.99999 )) THEN
          xi = 1.99999
        END IF
        cost = 1 - xi
        IF (( spin_effects )) THEN
          rejf=spin_rejection(qel,medium,elke,beta2,q1,cost,spin_index,.
     *    false.)
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          rnno = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          IF (( rnno .GT. rejf )) THEN
            GOTO 5110
          END IF
        END IF
        sint = sqrt(xi*(2-xi))
        return
      END IF
      write(i_log,*) ' '
      write(i_log,*) ' *************************************'
      write(i_log,*) ' Maximum step size in mscat exceeded! '
      write(i_log,*) ' Maximum step size initialized: 100000'
      write(i_log,*) ' Present lambda: ',lambda
      write(i_log,*) ' chia2: ',chia2
      write(i_log,*) ' q1 elke beta2: ',q1,elke,beta2
      write(i_log,*) ' medium: ',medium
      write(i_log,'(/a)') '***************** Error: '
      write(i_log,*) ' Stopping execution'
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
      end
      real*8 function spin_rejection(qel,medium,elke,beta2,q1,cost, spin
     *_index,is_single)
      implicit none
      real*8 elke,beta2,q1,cost
      integer*4 qel,medium
      logical spin_index,is_single
      common/spin_data/ spin_rej(1,0:1,0: 31,0:15,0:31), espin_min,espin
     *_max,espml,b2spin_min,b2spin_max, dbeta2,dbeta2i,dlener,dleneri,dq
     *q1,dqq1i, fool_intel_optimizer
      real*4 spin_rej,espin_min,espin_max,espml,b2spin_min,b2spin_max, d
     *beta2,dbeta2i,dlener,dleneri,dqq1,dqq1i
      logical fool_intel_optimizer
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      real*8 rnno,ai,qq1,aj,xi,ak
      integer*4 i,j,k
      save i,j
      IF (( spin_index )) THEN
        spin_index = .false.
        IF (( beta2 .GE. b2spin_min )) THEN
          ai = (beta2 - b2spin_min)*dbeta2i
          i = ai
          ai = ai - i
          i = i + 15 + 1
        ELSE IF(( elke .GT. espml )) THEN
          ai = (elke - espml)*dleneri
          i = ai
          ai = ai - i
        ELSE
          i = 0
          ai = -1
        END IF
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        rnno = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        IF((rnno .LT. ai))i = i + 1
        IF (( is_single )) THEN
          j = 0
        ELSE
          qq1 = 2*q1
          qq1 = qq1/(1 + qq1)
          aj = qq1*dqq1i
          j = aj
          IF (( j .GE. 15 )) THEN
            j = 15
          ELSE
            aj = aj - j
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            rnno = rng_array(rng_seed)
            rng_seed = rng_seed + 1
            IF((rnno .LT. aj))j = j + 1
          END IF
        END IF
      END IF
      xi = Sqrt(0.5*(1-cost))
      ak = xi*31
      k = ak
      ak = ak - k
      spin_rejection = (1-ak)*spin_rej(medium,qel,i,j,k) + ak*spin_rej(m
     *edium,qel,i,j,k+1)
      return
      end
      subroutine sscat(chia2,elke,beta2,qel,medium,spin_effects,cost,sin
     *t)
      implicit none
      real*8 chia2,elke,beta2,cost,sint
      integer*4 qel,medium
      logical spin_effects
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      real*8 xi,rnno,rejf,spin_rejection,qzero
      logical spin_index
      spin_index = .true.
5120  CONTINUE
      IF (( rng_seed .GT. 24 )) THEN
        call ranlux(rng_array)
        rng_seed = 1
      END IF
      xi = rng_array(rng_seed)
      rng_seed = rng_seed + 1
      xi = 2*chia2*xi/(1 - xi + chia2)
      cost = 1 - xi
      IF (( spin_effects )) THEN
        qzero=0
        rejf = spin_rejection(qel,medium,elke,beta2,qzero,cost,spin_inde
     *  x,.true.)
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        rnno = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        IF((rnno .GT. rejf))goto 5120
      END IF
      sint = sqrt(xi*(2 - xi))
      return
      end
      subroutine init_ms_SR
      implicit none
      common/ms_data/ ums_array(0:63,0:7,0:31), fms_array(0:63,0:7,0:31)
     *, wms_array(0:63,0:7,0:31), ims_array(0:63,0:7,0:31), llammin,llam
     *max,dllamb,dllambi,dqms,dqmsi
      real*4 ums_array,fms_array,wms_array, llammin,llammax,dllamb,dllam
     *bi,dqms,dqmsi
      integer*2 ims_array
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      integer*4 i,j,k
      write(i_log,'(/a,$)') 'Reading screened Rutherford MS data .......
     *........ '
      rewind(i_mscat)
      DO 5131 i=0,63
        DO 5141 j=0,7
          read(i_mscat,*) (ums_array(i,j,k),k=0,31)
          read(i_mscat,*) (fms_array(i,j,k),k=0,31)
          read(i_mscat,*) (wms_array(i,j,k),k=0,31-1)
          read(i_mscat,*) (ims_array(i,j,k),k=0,31-1)
          DO 5151 k=0,31-1
            fms_array(i,j,k) = fms_array(i,j,k+1)/fms_array(i,j,k)-1
            ims_array(i,j,k) = ims_array(i,j,k)-1
5151      CONTINUE
5152      CONTINUE
          fms_array(i,j,31)=fms_array(i,j,31-1)
5141    CONTINUE
5142    CONTINUE
5131  CONTINUE
5132  CONTINUE
      write(i_log,'(a)') ' done '
      llammin = Log(1.)
      llammax = Log(1e5)
      dllamb = (llammax-llammin)/63
      dllambi = 1./dllamb
      dqms = 0.5/7
      dqmsi = 1./dqms
      return
      end
      subroutine init_spin
      implicit none
      common/spin_data/ spin_rej(1,0:1,0: 31,0:15,0:31), espin_min,espin
     *_max,espml,b2spin_min,b2spin_max, dbeta2,dbeta2i,dlener,dleneri,dq
     *q1,dqq1i, fool_intel_optimizer
      real*4 spin_rej,espin_min,espin_max,espml,b2spin_min,b2spin_max, d
     *beta2,dbeta2i,dlener,dleneri,dqq1,dqq1i
      logical fool_intel_optimizer
      COMMON/ELECIN/ esig_e(1),psig_e(1), esige_max, psige_max, range_ep
     *(0:1,500,1), E_array(500,1), etae_ms0(500,1),etae_ms1(500,1),etap_
     *ms0(500,1),etap_ms1(500,1),q1ce_ms0(500,1),q1ce_ms1(500,1),q1cp_ms
     *0(500,1),q1cp_ms1(500,1),q2ce_ms0(500,1),q2ce_ms1(500,1),q2cp_ms0(
     *500,1),q2cp_ms1(500,1),blcce0(500,1),blcce1(500,1), EKE0(1),EKE1(1
     *), XR0(1),TEFF0(1),BLCC(1),XCC(1), ESIG0(500,1),ESIG1(500,1),PSIG0
     *(500,1),PSIG1(500,1),EDEDX0(500,1),EDEDX1(500,1),PDEDX0(500,1),PDE
     *DX1(500,1),EBR10(500,1),EBR11(500,1),PBR10(500,1),PBR11(500,1),PBR
     *20(500,1),PBR21(500,1),TMXS0(500,1),TMXS1(500,1), expeke1(1), IUNR
     *ST(1),EPSTFL(1),IAPRIM(1), sig_ismonotone(0:1,1)
      real*8 esig_e,   psig_e,   esige_max,  psige_max,  range_ep,  E_ar
     *ray,  etae_ms0,etae_ms1,  etap_ms0,etap_ms1,  q1ce_ms0,q1ce_ms1,
     *q1cp_ms0,q1cp_ms1,  q2ce_ms0,q2ce_ms1,  q2cp_ms0,q2cp_ms1,  blcce0
     *,blcce1,   expeke1,  EKE0,EKE1, XR0,  TEFF0,  BLCC,  XCC,  ESIG0,E
     *SIG1,  PSIG0,PSIG1,  EDEDX0,EDEDX1,  PDEDX0,PDEDX1,  EBR10,EBR11,
     * PBR10,PBR11,  PBR20,PBR21,  TMXS0,TMXS1
      integer*4 IUNRST,  EPSTFL,  IAPRIM
      logical sig_ismonotone
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      real*8 eta_array(0:1,0: 31), c_array(0:1,0: 31),g_array(0:1,0: 31)
     *, earray(0: 31),tmp_array(0: 31), sum_Z2,sum_Z,sum_A,sum_pz,Z,tmp,
     *Z23,g_m,g_r,sig,dedx, tau,tauc,beta2,eta,gamma,fmax, eil,e,si1e,si
     *2e,si1p,si2p,aae,etap, elarray(0: 31),farray(0: 31), af(0: 31),bf(
     *0: 31),cf(0: 31), df(0: 31),spline,dloge,eloge
      real*4 dum1,dum2,dum3,aux_o
      real*4 fmax_array(0:15)
      integer*2 i2_array(512),ii2
      integer*4 iq,i,j,k,i_ele,iii,iZ,iiZ,n_ener,n_q,n_point,je,neke, nd
     *ata,leil,length,ii4,irec
      character spin_file*256
      character*6 string
      integer*4 lnblnk1
      integer*4 spin_unit, rec_length, want_spin_unit
      integer egs_get_unit
      character data_version*32,endianess*4
      logical swap
      real*8 fine,TF_constant
      parameter (fine=137.03604,TF_constant=0.88534138)
      real*4 tmp_4
      character c_2(2), c_4(4)
      equivalence (ii2,c_2), (tmp_4,c_4)
      DO 5161 i=1,len(spin_file)
        spin_file(i:i) = ' '
5161  CONTINUE
5162  CONTINUE
      spin_file = hen_house(:lnblnk1(hen_house)) // 'data' // '/' // 'sp
     *inms.data'
      want_spin_unit = 61
      spin_unit = egs_get_unit(want_spin_unit)
      IF (( spin_unit .LT. 1 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'init_spin: failed to get a free fortran unit'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      rec_length = 276*4
      open(spin_unit,file=spin_file,form='unformatted',access='direct',
     *status='old',recl=rec_length,err=5170)
      read(spin_unit,rec=1,err=5180) data_version,endianess, espin_min,e
     *spin_max,b2spin_min,b2spin_max
      swap = endianess.ne.'1234'
      IF (( swap )) THEN
        tmp_4 = espin_min
        call egs_swap_4(c_4)
        espin_min = tmp_4
        tmp_4 = espin_max
        call egs_swap_4(c_4)
        espin_max = tmp_4
        tmp_4 = b2spin_min
        call egs_swap_4(c_4)
        b2spin_min = tmp_4
        tmp_4 = b2spin_max
        call egs_swap_4(c_4)
        b2spin_max = tmp_4
      END IF
      write(i_log,'(//a,a)') 'Reading spin data base from ',spin_file(:l
     *nblnk1(spin_file))
      write(i_log,'(a)') data_version
      write(i_log,'(a,a,a)') 'Data generated on a machine with ',endiane
     *ss, ' endianess'
      write(i_log,'(a,a)') 'The endianess of this CPU is ','1234'
      IF((swap))write(i_log,'(a)') '=> will need to do byte swaping'
      write(i_log,'(a,2f9.2,2f9.5,//)') 'Ranges: ',espin_min,espin_max,
     *b2spin_min,b2spin_max
      n_ener = 15
      n_q = 15
      n_point = 31
      dloge = log(espin_max/espin_min)/n_ener
      eloge = log(espin_min)
      earray(0) = espin_min
      IF (( fool_intel_optimizer )) THEN
        write(25,*) 'Energy grid:'
      END IF
      DO 5191 i=1,n_ener
        eloge = eloge + dloge
        earray(i) = exp(eloge)
        IF (( fool_intel_optimizer )) THEN
          write(25,*) i,earray(i)
        END IF
5191  CONTINUE
5192  CONTINUE
      dbeta2 = (b2spin_max - b2spin_min)/n_ener
      beta2 = b2spin_min
      earray(n_ener+1) = espin_max
      DO 5201 i=n_ener+2,2*n_ener+1
        beta2 = beta2 + dbeta2
        IF (( beta2 .LT. 0.999 )) THEN
          earray(i) = prm*1000.0*(1/sqrt(1-beta2)-1)
        ELSE
          earray(i) = 50585.1
        END IF
        IF (( fool_intel_optimizer )) THEN
          write(25,*) i,earray(i)
        END IF
5201  CONTINUE
5202  CONTINUE
      espin_min = espin_min/1000
      espin_max = espin_max/1000
      dlener = Log(espin_max/espin_min)/15
      dleneri = 1/dlener
      espml = Log(espin_min)
      dbeta2 = (b2spin_max-b2spin_min)/15
      dbeta2i = 1/dbeta2
      dqq1 = 0.5/15
      dqq1i = 1/dqq1
      DO 5211 medium=1,NMED
        write(i_log,'(a,i4,a,$)') '  medium ',medium,' .................
     *.... '
        DO 5221 iq=0,1
          DO 5231 i=0, 31
            eta_array(iq,i)=0
            c_array(iq,i)=0
            g_array(iq,i)=0
            DO 5241 j=0,15
              DO 5251 k=0,31
                spin_rej(medium,iq,i,j,k) = 0
5251          CONTINUE
5252          CONTINUE
5241        CONTINUE
5242        CONTINUE
5231      CONTINUE
5232      CONTINUE
5221    CONTINUE
5222    CONTINUE
        sum_Z2=0
        sum_A=0
        sum_pz=0
        sum_Z=0
        DO 5261 i_ele=1,NNE(medium)
          Z = ZELEM(medium,i_ele)
          iZ = int(Z+0.5)
          IF (( fool_intel_optimizer )) THEN
            write(25,*) ' Z = ',iZ
          END IF
          tmp = PZ(medium,i_ele)*Z*(Z+1)
          sum_Z2 = sum_Z2 + tmp
          sum_Z = sum_Z + PZ(medium,i_ele)*Z
          sum_A = sum_A + PZ(medium,i_ele)*WA(medium,i_ele)
          sum_pz = sum_pz + PZ(medium,i_ele)
          Z23 = Z**0.6666667
          DO 5271 iq=0,1
            DO 5281 i=0, 31
              irec = 1 + (iz-1)*4*(n_ener+1) + 2*iq*(n_ener+1) + i+1
              IF (( fool_intel_optimizer )) THEN
                write(25,*) '**** energy ',i,earray(i),irec
              END IF
              read(spin_unit,rec=irec,err=5180) dum1,dum2,dum3,aux_o,fma
     *        x_array,i2_array
              IF (( swap )) THEN
                tmp_4 = dum1
                call egs_swap_4(c_4)
                dum1 = tmp_4
                tmp_4 = dum2
                call egs_swap_4(c_4)
                dum2 = tmp_4
                tmp_4 = dum3
                call egs_swap_4(c_4)
                dum3 = tmp_4
                tmp_4 = aux_o
                call egs_swap_4(c_4)
                aux_o = tmp_4
              END IF
              eta_array(iq,i)=eta_array(iq,i)+tmp*Log(Z23*aux_o)
              tau = earray(i)/prm*0.001
              beta2 = tau*(tau+2)/(tau+1)**2
              eta = Z23/(fine*TF_constant)**2*aux_o/4/tau/(tau+2)
              c_array(iq,i)=c_array(iq,i)+ tmp*(Log(1+1/eta)-1/(1+eta))*
     *        dum1*dum3
              g_array(iq,i)=g_array(iq,i)+tmp*dum2
              DO 5291 j=0,15
                tmp_4 = fmax_array(j)
                IF((swap))call egs_swap_4(c_4)
                DO 5301 k=0,31
                  ii2 = i2_array((n_point+1)*j + k+1)
                  IF((swap))call egs_swap_2(c_2)
                  ii4 = ii2
                  IF((ii4 .LT. 0))ii4 = ii4 + 65536
                  dum1 = ii4
                  dum1 = dum1*tmp_4/65535
                  spin_rej(medium,iq,i,j,k) = spin_rej(medium,iq,i,j,k)
     *            + tmp*dum1
5301            CONTINUE
5302            CONTINUE
5291          CONTINUE
5292          CONTINUE
5281        CONTINUE
5282        CONTINUE
5271      CONTINUE
5272      CONTINUE
5261    CONTINUE
5262    CONTINUE
        DO 5311 iq=0,1
          DO 5321 i=0, 31
            DO 5331 j=0,15
              fmax = 0
              DO 5341 k=0,31
                IF (( spin_rej(medium,iq,i,j,k) .GT. fmax )) THEN
                  fmax = spin_rej(medium,iq,i,j,k)
                END IF
5341          CONTINUE
5342          CONTINUE
              DO 5351 k=0,31
                spin_rej(medium,iq,i,j,k) = spin_rej(medium,iq,i,j,k)/fm
     *          ax
5351          CONTINUE
5352          CONTINUE
5331        CONTINUE
5332        CONTINUE
5321      CONTINUE
5322      CONTINUE
5311    CONTINUE
5312    CONTINUE
        IF (( fool_intel_optimizer )) THEN
          write(25,*) 'Spin corrections as read in from file'
        END IF
        DO 5361 i=0, 31
          tau = earray(i)/prm*0.001
          beta2 = tau*(tau+2)/(tau+1)**2
          DO 5371 iq=0,1
            aux_o = Exp(eta_array(iq,i)/sum_Z2)/(fine*TF_constant)**2
            eta_array(iq,i) = 0.26112447*aux_o*blcc(medium)/xcc(medium)
            eta = aux_o/4/tau/(tau+2)
            gamma = 3*(1+eta)*(Log(1+1/eta)*(1+2*eta)-2)/ (Log(1+1/eta)*
     *      (1+eta)-1)
            g_array(iq,i) = g_array(iq,i)/sum_Z2/gamma
            c_array(iq,i) = c_array(iq,i)/sum_Z2/(Log(1+1/eta)-1/(1+eta)
     *      )
5371      CONTINUE
5372      CONTINUE
          IF (( fool_intel_optimizer )) THEN
            write(25,*) i,earray(i),eta_array(0,i),eta_array(1,i), c_arr
     *      ay(0,i),c_array(1,i),g_array(0,i),g_array(1,i)
          END IF
5361    CONTINUE
5362    CONTINUE
        eil = (1 - eke0(medium))/eke1(medium)
        e = Exp(eil)
        IF (( e .LE. espin_min )) THEN
          si1e = eta_array(0,0)
          si1p = eta_array(1,0)
        ELSE
          IF (( e .LE. espin_max )) THEN
            aae = (eil-espml)*dleneri
            je = aae
            aae = aae - je
          ELSE
            tau = e/prm
            beta2 = tau*(tau+2)/(tau+1)**2
            aae = (beta2 - b2spin_min)*dbeta2i
            je = aae
            aae = aae - je
            je = je + 15 + 1
          END IF
          si1e = (1-aae)*eta_array(0,je) + aae*eta_array(0,je+1)
          si1p = (1-aae)*eta_array(1,je) + aae*eta_array(1,je+1)
        END IF
        neke = meke(medium)
        IF (( fool_intel_optimizer )) THEN
          write(25,*) 'Interpolation table for eta correction'
        END IF
        DO 5381 i=1,neke - 1
          eil = (i+1 - eke0(medium))/eke1(medium)
          e = Exp(eil)
          IF (( e .LE. espin_min )) THEN
            si2e = eta_array(0,0)
            si2p = eta_array(1,0)
          ELSE
            IF (( e .LE. espin_max )) THEN
              aae = (eil-espml)*dleneri
              je = aae
              aae = aae - je
            ELSE
              tau = e/prm
              beta2 = tau*(tau+2)/(tau+1)**2
              aae = (beta2 - b2spin_min)*dbeta2i
              je = aae
              aae = aae - je
              je = je + 15 + 1
            END IF
            si2e = (1-aae)*eta_array(0,je) + aae*eta_array(0,je+1)
            si2p = (1-aae)*eta_array(1,je) + aae*eta_array(1,je+1)
          END IF
          etae_ms1(i,medium) = (si2e - si1e)*eke1(medium)
          etae_ms0(i,medium) = si2e - etae_ms1(i,medium)*eil
          etap_ms1(i,medium) = (si2p - si1p)*eke1(medium)
          etap_ms0(i,medium) = si2p - etap_ms1(i,medium)*eil
          IF (( fool_intel_optimizer )) THEN
            write(25,*) i,e,si2e,si2p,etae_ms1(i,medium), etae_ms0(i,med
     *      ium),etap_ms1(i,medium),etap_ms0(i,medium)
          END IF
          si1e = si2e
          si1p = si2p
5381    CONTINUE
5382    CONTINUE
        etae_ms1(neke,medium) = etae_ms1(neke-1,medium)
        etae_ms0(neke,medium) = etae_ms0(neke-1,medium)
        etap_ms1(neke,medium) = etap_ms1(neke-1,medium)
        etap_ms0(neke,medium) = etap_ms0(neke-1,medium)
        IF (( fool_intel_optimizer )) THEN
          write(25,*) 'elarray:'
        END IF
        DO 5391 i=0,15
          elarray(i) = Log(earray(i)/1000)
          farray(i) = c_array(0,i)
          IF (( fool_intel_optimizer )) THEN
            write(25,*) elarray(i),earray(i)
          END IF
5391    CONTINUE
5392    CONTINUE
        DO 5401 i=15+1, 31-1
          elarray(i) = Log(earray(i+1)/1000)
          farray(i) = c_array(0,i+1)
          IF (( fool_intel_optimizer )) THEN
            write(25,*) elarray(i),earray(i+1)
          END IF
5401    CONTINUE
5402    CONTINUE
        ndata =  31+1
        IF (( ue(medium) .GT. 1e5 )) THEN
          elarray(ndata-1) = Log(ue(medium))
        ELSE
          elarray(ndata-1) = Log(1e5)
        END IF
        farray(ndata-1) = 1
        call set_spline(elarray,farray,af,bf,cf,df,ndata)
        eil = (1 - eke0(medium))/eke1(medium)
        si1e = spline(eil,elarray,af,bf,cf,df,ndata)
        IF (( fool_intel_optimizer )) THEN
          write(25,*) 'Interpolation table for q1 correction (e-)'
        END IF
        DO 5411 i=1,neke-1
          eil = (i+1 - eke0(medium))/eke1(medium)
          si2e = spline(eil,elarray,af,bf,cf,df,ndata)
          q1ce_ms1(i,medium) = (si2e - si1e)*eke1(medium)
          q1ce_ms0(i,medium) = si2e - q1ce_ms1(i,medium)*eil
          IF (( fool_intel_optimizer )) THEN
            write(25,*) Exp(eil),si2e,q1ce_ms1(i,medium), q1ce_ms0(i,med
     *      ium)
          END IF
          si1e = si2e
5411    CONTINUE
5412    CONTINUE
        q1ce_ms1(neke,medium) = q1ce_ms1(neke-1,medium)
        q1ce_ms0(neke,medium) = q1ce_ms0(neke-1,medium)
        IF (( fool_intel_optimizer )) THEN
          write(25,*) 'Postrons:'
        END IF
        DO 5421 i=0,15
          farray(i) = c_array(1,i)
5421    CONTINUE
5422    CONTINUE
        DO 5431 i=15+1, 31-1
          farray(i) = c_array(1,i+1)
5431    CONTINUE
5432    CONTINUE
        call set_spline(elarray,farray,af,bf,cf,df,ndata)
        eil = (1 - eke0(medium))/eke1(medium)
        si1e = spline(eil,elarray,af,bf,cf,df,ndata)
        IF (( fool_intel_optimizer )) THEN
          write(25,*) 'Interpolation table for q1 correction (e+)'
        END IF
        DO 5441 i=1,neke-1
          eil = (i+1 - eke0(medium))/eke1(medium)
          si2e = spline(eil,elarray,af,bf,cf,df,ndata)
          q1cp_ms1(i,medium) = (si2e - si1e)*eke1(medium)
          q1cp_ms0(i,medium) = si2e - q1cp_ms1(i,medium)*eil
          IF (( fool_intel_optimizer )) THEN
            write(25,*) Exp(eil),si2e,q1cp_ms1(i,medium), q1cp_ms0(i,med
     *      ium)
          END IF
          si1e = si2e
5441    CONTINUE
5442    CONTINUE
        q1cp_ms1(neke,medium) = q1cp_ms1(neke-1,medium)
        q1cp_ms0(neke,medium) = q1cp_ms0(neke-1,medium)
        DO 5451 i=0,15
          farray(i) = g_array(0,i)
5451    CONTINUE
5452    CONTINUE
        DO 5461 i=15+1, 31-1
          farray(i) = g_array(0,i+1)
5461    CONTINUE
5462    CONTINUE
        call set_spline(elarray,farray,af,bf,cf,df,ndata)
        eil = (1 - eke0(medium))/eke1(medium)
        si1e = spline(eil,elarray,af,bf,cf,df,ndata)
        IF (( fool_intel_optimizer )) THEN
          write(25,*) 'Interpolation table for q2 correction (e-)'
        END IF
        DO 5471 i=1,neke-1
          eil = (i+1 - eke0(medium))/eke1(medium)
          si2e = spline(eil,elarray,af,bf,cf,df,ndata)
          q2ce_ms1(i,medium) = (si2e - si1e)*eke1(medium)
          q2ce_ms0(i,medium) = si2e - q2ce_ms1(i,medium)*eil
          IF (( fool_intel_optimizer )) THEN
            write(25,*) Exp(eil),si2e,q2ce_ms1(i,medium), q2ce_ms0(i,med
     *      ium)
          END IF
          si1e = si2e
5471    CONTINUE
5472    CONTINUE
        q2ce_ms1(neke,medium) = q2ce_ms1(neke-1,medium)
        q2ce_ms0(neke,medium) = q2ce_ms0(neke-1,medium)
        DO 5481 i=0,15
          farray(i) = g_array(1,i)
5481    CONTINUE
5482    CONTINUE
        DO 5491 i=15+1, 31-1
          farray(i) = g_array(1,i+1)
5491    CONTINUE
5492    CONTINUE
        call set_spline(elarray,farray,af,bf,cf,df,ndata)
        eil = (1 - eke0(medium))/eke1(medium)
        si1e = spline(eil,elarray,af,bf,cf,df,ndata)
        IF (( fool_intel_optimizer )) THEN
          write(25,*) 'Interpolation table for q2 correction (e+)'
        END IF
        DO 5501 i=1,neke-1
          eil = (i+1 - eke0(medium))/eke1(medium)
          si2e = spline(eil,elarray,af,bf,cf,df,ndata)
          q2cp_ms1(i,medium) = (si2e - si1e)*eke1(medium)
          q2cp_ms0(i,medium) = si2e - q2cp_ms1(i,medium)*eil
          IF (( fool_intel_optimizer )) THEN
            write(25,*) Exp(eil),si2e,q2cp_ms1(i,medium), q2cp_ms0(i,med
     *      ium)
          END IF
          si1e = si2e
5501    CONTINUE
5502    CONTINUE
        q2cp_ms1(neke,medium) = q2cp_ms1(neke-1,medium)
        q2cp_ms0(neke,medium) = q2cp_ms0(neke-1,medium)
        tauc = te(medium)/prm
        si1e = 1
        DO 5511 i=1,neke-1
          eil = (i+1 - eke0(medium))/eke1(medium)
          e = Exp(eil)
          leil=i+1
          tau=e/prm
          IF (( tau .GT. 2*tauc )) THEN
            sig=esig1(Leil,MEDIUM)*eil+esig0(Leil,MEDIUM)
            dedx=ededx1(Leil,MEDIUM)*eil+ededx0(Leil,MEDIUM)
            sig = sig/dedx
            IF (( sig .GT. 1e-6 )) THEN
              etap=etae_ms1(Leil,MEDIUM)*eil+etae_ms0(Leil,MEDIUM)
              eta = 0.25*etap*xcc(medium)/blcc(medium)/tau/(tau+2)
              g_r = (1+2*eta)*Log(1+1/eta)-2
              g_m = Log(0.5*tau/tauc)+ (1+((tau+2)/(tau+1))**2)*Log(2*(t
     *        au-tauc+2)/(tau+4))- 0.25*(tau+2)*(tau+2+2*(2*tau+1)/(tau+
     *        1)**2)* Log((tau+4)*(tau-tauc)/tau/(tau-tauc+2))+ 0.5*(tau
     *        -2*tauc)*(tau+2)*(1/(tau-tauc)-1/(tau+1)**2)
              IF (( g_m .LT. g_r )) THEN
                g_m = g_m/g_r
              ELSE
                g_m = 1
              END IF
              si2e = 1 - g_m*sum_Z/sum_Z2
            ELSE
              si2e = 1
            END IF
          ELSE
            si2e = 1
          END IF
          blcce1(i,medium) = (si2e - si1e)*eke1(medium)
          blcce0(i,medium) = si2e - blcce1(i,medium)*eil
          si1e = si2e
5511    CONTINUE
5512    CONTINUE
        blcce1(neke,medium) = blcce1(neke-1,medium)
        blcce0(neke,medium) = blcce0(neke-1,medium)
        write(i_log,'(a)') ' done'
5211  CONTINUE
5212  CONTINUE
      close(spin_unit)
      return
5170  write(i_log,'(/a)') '***************** Error: '
      write(i_log,'(a,a)') 'Failed to open spin data file ',spin_file(:l
     *nblnk1(spin_file))
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
5180  write(i_log,'(/a)') '***************** Error: '
      write(i_log,*) 'Error while reading spin data file for element',iZ
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
      return
      end
      subroutine init_spin_old
      implicit none
      common/spin_data/ spin_rej(1,0:1,0: 31,0:15,0:31), espin_min,espin
     *_max,espml,b2spin_min,b2spin_max, dbeta2,dbeta2i,dlener,dleneri,dq
     *q1,dqq1i, fool_intel_optimizer
      real*4 spin_rej,espin_min,espin_max,espml,b2spin_min,b2spin_max, d
     *beta2,dbeta2i,dlener,dleneri,dqq1,dqq1i
      logical fool_intel_optimizer
      COMMON/ELECIN/ esig_e(1),psig_e(1), esige_max, psige_max, range_ep
     *(0:1,500,1), E_array(500,1), etae_ms0(500,1),etae_ms1(500,1),etap_
     *ms0(500,1),etap_ms1(500,1),q1ce_ms0(500,1),q1ce_ms1(500,1),q1cp_ms
     *0(500,1),q1cp_ms1(500,1),q2ce_ms0(500,1),q2ce_ms1(500,1),q2cp_ms0(
     *500,1),q2cp_ms1(500,1),blcce0(500,1),blcce1(500,1), EKE0(1),EKE1(1
     *), XR0(1),TEFF0(1),BLCC(1),XCC(1), ESIG0(500,1),ESIG1(500,1),PSIG0
     *(500,1),PSIG1(500,1),EDEDX0(500,1),EDEDX1(500,1),PDEDX0(500,1),PDE
     *DX1(500,1),EBR10(500,1),EBR11(500,1),PBR10(500,1),PBR11(500,1),PBR
     *20(500,1),PBR21(500,1),TMXS0(500,1),TMXS1(500,1), expeke1(1), IUNR
     *ST(1),EPSTFL(1),IAPRIM(1), sig_ismonotone(0:1,1)
      real*8 esig_e,   psig_e,   esige_max,  psige_max,  range_ep,  E_ar
     *ray,  etae_ms0,etae_ms1,  etap_ms0,etap_ms1,  q1ce_ms0,q1ce_ms1,
     *q1cp_ms0,q1cp_ms1,  q2ce_ms0,q2ce_ms1,  q2cp_ms0,q2cp_ms1,  blcce0
     *,blcce1,   expeke1,  EKE0,EKE1, XR0,  TEFF0,  BLCC,  XCC,  ESIG0,E
     *SIG1,  PSIG0,PSIG1,  EDEDX0,EDEDX1,  PDEDX0,PDEDX1,  EBR10,EBR11,
     * PBR10,PBR11,  PBR20,PBR21,  TMXS0,TMXS1
      integer*4 IUNRST,  EPSTFL,  IAPRIM
      logical sig_ismonotone
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      real*8 eta_array(0:1,0: 31), c_array(0:1,0: 31),g_array(0:1,0: 31)
     *, earray(0: 31),tmp_array(0: 31), sum_Z2,sum_Z,sum_A,sum_pz,Z,tmp,
     *Z23,g_m,g_r,sig,dedx, dum1,dum2,dum3,aux_o,tau,tauc,beta2,eta,gamm
     *a,fmax, eil,e,si1e,si2e,si1p,si2p,aae,etap, elarray(0: 31),farray(
     *0: 31), af(0: 31),bf(0: 31),cf(0: 31), df(0: 31),spline
      integer*4 iq,i,j,k,i_ele,iii,iZ,iiZ,n_ener,n_q,n_point,je,neke, nd
     *ata,leil,length,want_spin_unit,spin_unit,egs_get_unit
      character spin_file*256
      character*6 string
      integer*4 lnblnk1
      real*8 fine,TF_constant
      parameter (fine=137.03604,TF_constant=0.88534138)
      DO 5521 i=1,len(spin_file)
        spin_file(i:i) = ' '
5521  CONTINUE
5522  CONTINUE
      spin_file = hen_house(:lnblnk1(hen_house)) // 'data' // '/' // 'sp
     *inms' // '/' // 'z000'
      length = lnblnk1(spin_file)
      DO 5531 medium=1,NMED
        write(i_log,'(a,i4,a,$)') '  Initializing spin data for medium '
     *  ,medium, ' ..................... '
        DO 5541 iq=0,1
          DO 5551 i=0, 31
            eta_array(iq,i)=0
            c_array(iq,i)=0
            g_array(iq,i)=0
            DO 5561 j=0,15
              DO 5571 k=0,31
                spin_rej(medium,iq,i,j,k) = 0
5571          CONTINUE
5572          CONTINUE
5561        CONTINUE
5562        CONTINUE
5551      CONTINUE
5552      CONTINUE
5541    CONTINUE
5542    CONTINUE
        sum_Z2=0
        sum_A=0
        sum_pz=0
        sum_Z=0
        DO 5581 i_ele=1,NNE(medium)
          Z = ZELEM(medium,i_ele)
          iZ = int(Z+0.5)
          tmp = PZ(medium,i_ele)*Z*(Z+1)
          iii = iZ/100
          spin_file(length-2:length-2) = char(iii+48)
          iiZ = iZ - iii*100
          iii = iiZ/10
          spin_file(length-1:length-1) = char(iii+48)
          iiZ = iiZ - 10*iii
          spin_file(length:length) = char(iiZ+48)
          want_spin_unit = 61
          spin_unit = egs_get_unit(want_spin_unit)
          IF (( spin_unit .LT. 1 )) THEN
            write(i_log,'(/a)') '***************** Error: '
            write(i_log,*) 'init_spin: failed to get a free fortran unit
     *'
            write(i_log,'(/a)') '***************** Quiting now.'
            call exit(1)
          END IF
          open(spin_unit,file=spin_file,status='old',err=5590)
          read(spin_unit,*) espin_min,espin_max,b2spin_min,b2spin_max
          read(spin_unit,*) n_ener,n_q,n_point
          IF (( n_ener .NE. 15 .OR. n_q .NE. 15 .OR. n_point .NE. 31)) T
     *    HEN
            write(i_log,'(/a)') '***************** Error: '
            write(i_log,*) ' Wrong spin file for Z = ',iZ
            write(i_log,'(/a)') '***************** Quiting now.'
            call exit(1)
          END IF
          sum_Z2 = sum_Z2 + tmp
          sum_Z = sum_Z + PZ(medium,i_ele)*Z
          sum_A = sum_A + PZ(medium,i_ele)*WA(medium,i_ele)
          sum_pz = sum_pz + PZ(medium,i_ele)
          Z23 = Z**0.6666667
          DO 5601 iq=0,1
            read(spin_unit,*)
            read(spin_unit,*)
            DO 5611 i=0, 31
              read(spin_unit,'(a,g14.6)') string,earray(i)
              read(spin_unit,*) dum1,dum2,dum3,aux_o
              eta_array(iq,i)=eta_array(iq,i)+tmp*Log(Z23*aux_o)
              tau = earray(i)/prm*0.001
              beta2 = tau*(tau+2)/(tau+1)**2
              eta = Z23/(fine*TF_constant)**2*aux_o/4/tau/(tau+2)
              c_array(iq,i)=c_array(iq,i)+ tmp*(Log(1+1/eta)-1/(1+eta))*
     *        dum1*dum3
              g_array(iq,i)=g_array(iq,i)+tmp*dum2
              DO 5621 j=0,15
                read(spin_unit,*) tmp_array
                DO 5631 k=0,31
                  spin_rej(medium,iq,i,j,k) = spin_rej(medium,iq,i,j,k)
     *            + tmp*tmp_array(k)
5631            CONTINUE
5632            CONTINUE
5621          CONTINUE
5622          CONTINUE
5611        CONTINUE
5612        CONTINUE
5601      CONTINUE
5602      CONTINUE
          close(spin_unit)
5581    CONTINUE
5582    CONTINUE
        DO 5641 iq=0,1
          DO 5651 i=0, 31
            DO 5661 j=0,15
              fmax = 0
              DO 5671 k=0,31
                IF (( spin_rej(medium,iq,i,j,k) .GT. fmax )) THEN
                  fmax = spin_rej(medium,iq,i,j,k)
                END IF
5671          CONTINUE
5672          CONTINUE
              DO 5681 k=0,31
                spin_rej(medium,iq,i,j,k) = spin_rej(medium,iq,i,j,k)/fm
     *          ax
5681          CONTINUE
5682          CONTINUE
5661        CONTINUE
5662        CONTINUE
5651      CONTINUE
5652      CONTINUE
5641    CONTINUE
5642    CONTINUE
        DO 5691 i=0, 31
          tau = earray(i)/prm*0.001
          beta2 = tau*(tau+2)/(tau+1)**2
          DO 5701 iq=0,1
            aux_o = Exp(eta_array(iq,i)/sum_Z2)/(fine*TF_constant)**2
            eta_array(iq,i) = 0.26112447*aux_o*blcc(medium)/xcc(medium)
            eta = aux_o/4/tau/(tau+2)
            gamma = 3*(1+eta)*(Log(1+1/eta)*(1+2*eta)-2)/ (Log(1+1/eta)*
     *      (1+eta)-1)
            g_array(iq,i) = g_array(iq,i)/sum_Z2/gamma
            c_array(iq,i) = c_array(iq,i)/sum_Z2/(Log(1+1/eta)-1/(1+eta)
     *      )
5701      CONTINUE
5702      CONTINUE
5691    CONTINUE
5692    CONTINUE
        espin_min = espin_min/1000
        espin_max = espin_max/1000
        dlener = Log(espin_max/espin_min)/15
        dleneri = 1/dlener
        espml = Log(espin_min)
        dbeta2 = (b2spin_max-b2spin_min)/15
        dbeta2i = 1/dbeta2
        dqq1 = 0.5/15
        dqq1i = 1/dqq1
        eil = (1 - eke0(medium))/eke1(medium)
        e = Exp(eil)
        IF (( e .LE. espin_min )) THEN
          si1e = eta_array(0,0)
          si1p = eta_array(1,0)
        ELSE
          IF (( e .LE. espin_max )) THEN
            aae = (eil-espml)*dleneri
            je = aae
            aae = aae - je
          ELSE
            tau = e/prm
            beta2 = tau*(tau+2)/(tau+1)**2
            aae = (beta2 - b2spin_min)*dbeta2i
            je = aae
            aae = aae - je
            je = je + 15 + 1
          END IF
          si1e = (1-aae)*eta_array(0,je) + aae*eta_array(0,je+1)
          si1p = (1-aae)*eta_array(1,je) + aae*eta_array(1,je+1)
        END IF
        neke = meke(medium)
        DO 5711 i=1,neke - 1
          eil = (i+1 - eke0(medium))/eke1(medium)
          e = Exp(eil)
          IF (( e .LE. espin_min )) THEN
            si2e = eta_array(0,0)
            si2p = eta_array(1,0)
          ELSE
            IF (( e .LE. espin_max )) THEN
              aae = (eil-espml)*dleneri
              je = aae
              aae = aae - je
            ELSE
              tau = e/prm
              beta2 = tau*(tau+2)/(tau+1)**2
              aae = (beta2 - b2spin_min)*dbeta2i
              je = aae
              aae = aae - je
              je = je + 15 + 1
            END IF
            si2e = (1-aae)*eta_array(0,je) + aae*eta_array(0,je+1)
            si2p = (1-aae)*eta_array(1,je) + aae*eta_array(1,je+1)
          END IF
          etae_ms1(i,medium) = (si2e - si1e)*eke1(medium)
          etae_ms0(i,medium) = si2e - etae_ms1(i,medium)*eil
          etap_ms1(i,medium) = (si2p - si1p)*eke1(medium)
          etap_ms0(i,medium) = si2p - etap_ms1(i,medium)*eil
          si1e = si2e
          si1p = si2p
5711    CONTINUE
5712    CONTINUE
        etae_ms1(neke,medium) = etae_ms1(neke-1,medium)
        etae_ms0(neke,medium) = etae_ms0(neke-1,medium)
        etap_ms1(neke,medium) = etap_ms1(neke-1,medium)
        etap_ms0(neke,medium) = etap_ms0(neke-1,medium)
        DO 5721 i=0,15
          elarray(i) = Log(earray(i)/1000)
          farray(i) = c_array(0,i)
5721    CONTINUE
5722    CONTINUE
        DO 5731 i=15+1, 31-1
          elarray(i) = Log(earray(i+1)/1000)
          farray(i) = c_array(0,i+1)
5731    CONTINUE
5732    CONTINUE
        ndata =  31+1
        IF (( ue(medium) .GT. 1e5 )) THEN
          elarray(ndata-1) = Log(ue(medium))
        ELSE
          elarray(ndata-1) = Log(1e5)
        END IF
        farray(ndata-1) = 1
        call set_spline(elarray,farray,af,bf,cf,df,ndata)
        eil = (1 - eke0(medium))/eke1(medium)
        si1e = spline(eil,elarray,af,bf,cf,df,ndata)
        DO 5741 i=1,neke-1
          eil = (i+1 - eke0(medium))/eke1(medium)
          si2e = spline(eil,elarray,af,bf,cf,df,ndata)
          q1ce_ms1(i,medium) = (si2e - si1e)*eke1(medium)
          q1ce_ms0(i,medium) = si2e - q1ce_ms1(i,medium)*eil
          si1e = si2e
5741    CONTINUE
5742    CONTINUE
        q1ce_ms1(neke,medium) = q1ce_ms1(neke-1,medium)
        q1ce_ms0(neke,medium) = q1ce_ms0(neke-1,medium)
        DO 5751 i=0,15
          farray(i) = c_array(1,i)
5751    CONTINUE
5752    CONTINUE
        DO 5761 i=15+1, 31-1
          farray(i) = c_array(1,i+1)
5761    CONTINUE
5762    CONTINUE
        call set_spline(elarray,farray,af,bf,cf,df,ndata)
        eil = (1 - eke0(medium))/eke1(medium)
        si1e = spline(eil,elarray,af,bf,cf,df,ndata)
        DO 5771 i=1,neke-1
          eil = (i+1 - eke0(medium))/eke1(medium)
          si2e = spline(eil,elarray,af,bf,cf,df,ndata)
          q1cp_ms1(i,medium) = (si2e - si1e)*eke1(medium)
          q1cp_ms0(i,medium) = si2e - q1cp_ms1(i,medium)*eil
          si1e = si2e
5771    CONTINUE
5772    CONTINUE
        q1cp_ms1(neke,medium) = q1cp_ms1(neke-1,medium)
        q1cp_ms0(neke,medium) = q1cp_ms0(neke-1,medium)
        DO 5781 i=0,15
          farray(i) = g_array(0,i)
5781    CONTINUE
5782    CONTINUE
        DO 5791 i=15+1, 31-1
          farray(i) = g_array(0,i+1)
5791    CONTINUE
5792    CONTINUE
        call set_spline(elarray,farray,af,bf,cf,df,ndata)
        eil = (1 - eke0(medium))/eke1(medium)
        si1e = spline(eil,elarray,af,bf,cf,df,ndata)
        DO 5801 i=1,neke-1
          eil = (i+1 - eke0(medium))/eke1(medium)
          si2e = spline(eil,elarray,af,bf,cf,df,ndata)
          q2ce_ms1(i,medium) = (si2e - si1e)*eke1(medium)
          q2ce_ms0(i,medium) = si2e - q2ce_ms1(i,medium)*eil
          si1e = si2e
5801    CONTINUE
5802    CONTINUE
        q2ce_ms1(neke,medium) = q2ce_ms1(neke-1,medium)
        q2ce_ms0(neke,medium) = q2ce_ms0(neke-1,medium)
        DO 5811 i=0,15
          farray(i) = g_array(1,i)
5811    CONTINUE
5812    CONTINUE
        DO 5821 i=15+1, 31-1
          farray(i) = g_array(1,i+1)
5821    CONTINUE
5822    CONTINUE
        call set_spline(elarray,farray,af,bf,cf,df,ndata)
        eil = (1 - eke0(medium))/eke1(medium)
        si1e = spline(eil,elarray,af,bf,cf,df,ndata)
        DO 5831 i=1,neke-1
          eil = (i+1 - eke0(medium))/eke1(medium)
          si2e = spline(eil,elarray,af,bf,cf,df,ndata)
          q2cp_ms1(i,medium) = (si2e - si1e)*eke1(medium)
          q2cp_ms0(i,medium) = si2e - q2cp_ms1(i,medium)*eil
5831    CONTINUE
5832    CONTINUE
        q2cp_ms1(neke,medium) = q2cp_ms1(neke-1,medium)
        q2cp_ms0(neke,medium) = q2cp_ms0(neke-1,medium)
        tauc = te(medium)/prm
        si1e = 1
        DO 5841 i=1,neke-1
          eil = (i+1 - eke0(medium))/eke1(medium)
          e = Exp(eil)
          leil=i+1
          tau=e/prm
          IF (( tau .GT. 2*tauc )) THEN
            sig=esig1(Leil,MEDIUM)*eil+esig0(Leil,MEDIUM)
            dedx=ededx1(Leil,MEDIUM)*eil+ededx0(Leil,MEDIUM)
            sig = sig/dedx
            IF (( sig .GT. 1e-6 )) THEN
              etap=etae_ms1(Leil,MEDIUM)*eil+etae_ms0(Leil,MEDIUM)
              eta = 0.25*etap*xcc(medium)/blcc(medium)/tau/(tau+2)
              g_r = (1+2*eta)*Log(1+1/eta)-2
              g_m = Log(0.5*tau/tauc)+ (1+((tau+2)/(tau+1))**2)*Log(2*(t
     *        au-tauc+2)/(tau+4))- 0.25*(tau+2)*(tau+2+2*(2*tau+1)/(tau+
     *        1)**2)* Log((tau+4)*(tau-tauc)/tau/(tau-tauc+2))+ 0.5*(tau
     *        -2*tauc)*(tau+2)*(1/(tau-tauc)-1/(tau+1)**2)
              IF (( g_m .LT. g_r )) THEN
                g_m = g_m/g_r
              ELSE
                g_m = 1
              END IF
              si2e = 1 - g_m*sum_Z/sum_Z2
            ELSE
              si2e = 1
            END IF
          ELSE
            si2e = 1
          END IF
          blcce1(i,medium) = (si2e - si1e)*eke1(medium)
          blcce0(i,medium) = si2e - blcce1(i,medium)*eil
          si1e = si2e
5841    CONTINUE
5842    CONTINUE
        blcce1(neke,medium) = blcce1(neke-1,medium)
        blcce0(neke,medium) = blcce0(neke-1,medium)
        write(i_log,'(a)') ' done'
5531  CONTINUE
5532  CONTINUE
      return
5590  write(i_log,*) ' ******************** Error in init_spin *********
     *********** '
      write(i_log,'(a,a)') '  could not open file ',spin_file
      write(i_log,'(/a)') '***************** Error: '
      write(i_log,*) ' terminating execution '
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
      end
      subroutine msdist_pII ( e0,eloss,tustep,rhof,med,qel,spin_effects,
     *u0,v0,w0,x0,y0,z0,  us,vs,ws,xf,yf,zf,ustep )
      implicit none
      real*8 e0,  eloss,  rhof,  tustep,  u0,  v0,  w0,  x0,  y0,  z0
      integer*4 med, qel
      logical spin_effects
      real*8 us,  vs,  ws,  xf,  yf,  zf,  ustep
      real*8 b,  blccc,  xcccc,  c,  eta,eta1,  chia2,  chilog,  cphi0,
     *  cphi1,  cphi2,  w1,  w2,  w1v2,  delta,  e,  elke,  beta2,  etap
     *,  xi_corr,  ms_corr, tau,  tau2,  epsilon,  epsilonp,  temp,temp1
     *, temp2,  factor,  gamma,  lambda,   p2,  p2i,  q1,  rhophi2,  sin
     *t0,  sint02,  sint0i,  sint1,  sint2,  sphi0,   sphi1,  sphi2,  u2
     *p,  u2,  v2,  ut,  vt,  wt,  xi,  xphi,  xphi2,  yphi,  yphi2
      logical find_index,  spin_index
      integer*4 lelke
      COMMON/ELECIN/ esig_e(1),psig_e(1), esige_max, psige_max, range_ep
     *(0:1,500,1), E_array(500,1), etae_ms0(500,1),etae_ms1(500,1),etap_
     *ms0(500,1),etap_ms1(500,1),q1ce_ms0(500,1),q1ce_ms1(500,1),q1cp_ms
     *0(500,1),q1cp_ms1(500,1),q2ce_ms0(500,1),q2ce_ms1(500,1),q2cp_ms0(
     *500,1),q2cp_ms1(500,1),blcce0(500,1),blcce1(500,1), EKE0(1),EKE1(1
     *), XR0(1),TEFF0(1),BLCC(1),XCC(1), ESIG0(500,1),ESIG1(500,1),PSIG0
     *(500,1),PSIG1(500,1),EDEDX0(500,1),EDEDX1(500,1),PDEDX0(500,1),PDE
     *DX1(500,1),EBR10(500,1),EBR11(500,1),PBR10(500,1),PBR11(500,1),PBR
     *20(500,1),PBR21(500,1),TMXS0(500,1),TMXS1(500,1), expeke1(1), IUNR
     *ST(1),EPSTFL(1),IAPRIM(1), sig_ismonotone(0:1,1)
      real*8 esig_e,   psig_e,   esige_max,  psige_max,  range_ep,  E_ar
     *ray,  etae_ms0,etae_ms1,  etap_ms0,etap_ms1,  q1ce_ms0,q1ce_ms1,
     *q1cp_ms0,q1cp_ms1,  q2ce_ms0,q2ce_ms1,  q2cp_ms0,q2cp_ms1,  blcce0
     *,blcce1,   expeke1,  EKE0,EKE1, XR0,  TEFF0,  BLCC,  XCC,  ESIG0,E
     *SIG1,  PSIG0,PSIG1,  EDEDX0,EDEDX1,  PDEDX0,PDEDX1,  EBR10,EBR11,
     * PBR10,PBR11,  PBR20,PBR21,  TMXS0,TMXS1
      integer*4 IUNRST,  EPSTFL,  IAPRIM
      logical sig_ismonotone
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/UPHIOT/THETA,SINTHE,COSTHE,SINPHI, COSPHI,PI,TWOPI,PI5D2
      real*8 THETA,  SINTHE,  COSTHE,  SINPHI,  COSPHI,  PI,TWOPI,PI5D2
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      common/CH_steps/ count_pII_steps,count_all_steps,is_ch_step
      real*8 count_pII_steps,count_all_steps
      logical is_ch_step
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common/emf_inputs/ExIN,EyIN,EzIN,  EMLMTIN,  BxIN, ByIN, BzIN,  Bx
     *, By, Bz,  Bx_new, By_new, Bz_new,  emfield_on
      real*8 ExIN,EyIN,EzIN, EMLMTIN, BxIN,ByIN,BzIN, Bx,By,Bz, Bx_new,B
     *y_new,Bz_new
      logical emfield_on
      medium = med
      count_pII_steps = count_pII_steps + 1
      blccc = blcc(medium)
      xcccc = xcc(medium)
      e = e0 - 0.5*eloss
      tau = e/prm
      tau2 = tau*tau
      epsilon = eloss/e0
      epsilonp= eloss/e
      e = e * (1 - epsilonp*epsilonp*(6+10*tau+5*tau2)/(24*tau2+72*tau+4
     *8))
      p2 = e*(e + rmt2)
      beta2 = p2/(p2 + rmsq)
      chia2 = xcccc/(4*p2*blccc)
      lambda = 0.5*tustep*rhof*blccc/beta2
      temp2 = 0.166666*(4+tau*(6+tau*(7+tau*(4+tau))))* (epsilonp/((tau+
     *1)*(tau+2)))**2
      lambda = lambda*(1 - temp2)
      IF (( spin_effects )) THEN
        elke = Log(e)
        Lelke=eke1(MEDIUM)*elke+eke0(MEDIUM)
        IF (( lelke .LT. 1 )) THEN
          lelke = 1
          elke = (1 - eke0(medium))/eke1(medium)
        END IF
        IF (( qel .EQ. 0 )) THEN
          etap=etae_ms1(Lelke,MEDIUM)*elke+etae_ms0(Lelke,MEDIUM)
          xi_corr=q1ce_ms1(Lelke,MEDIUM)*elke+q1ce_ms0(Lelke,MEDIUM)
          gamma=q2ce_ms1(Lelke,MEDIUM)*elke+q2ce_ms0(Lelke,MEDIUM)
        ELSE
          etap=etap_ms1(Lelke,MEDIUM)*elke+etap_ms0(Lelke,MEDIUM)
          xi_corr=q1cp_ms1(Lelke,MEDIUM)*elke+q1cp_ms0(Lelke,MEDIUM)
          gamma=q2cp_ms1(Lelke,MEDIUM)*elke+q2cp_ms0(Lelke,MEDIUM)
        END IF
        ms_corr=blcce1(Lelke,MEDIUM)*elke+blcce0(Lelke,MEDIUM)
      ELSE
        etap = 1
        xi_corr = 1
        gamma = 1
        ms_corr = 1
      END IF
      chia2 = chia2*etap
      lambda = lambda/(etap*(1+chia2))*ms_corr
      chilog = Log(1 + 1/chia2)
      q1 = 2*chia2*(chilog*(1 + chia2) - 1)
      gamma = 6*chia2*(1 + chia2)*(chilog*(1 + 2*chia2) - 2)/q1*gamma
      xi = q1*lambda
      find_index = .true.
      spin_index = .true.
      call mscat(lambda,chia2,xi,elke,beta2,qel,medium, spin_effects,fin
     *d_index,spin_index, w1,sint1)
5851  CONTINUE
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        xphi = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        xphi = 2*xphi - 1
        xphi2 = xphi*xphi
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        yphi = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        yphi2 = yphi*yphi
        rhophi2 = xphi2 + yphi2
        IF(rhophi2.LE.1)GO TO5852
      GO TO 5851
5852  CONTINUE
      rhophi2 = 1/rhophi2
      cphi1 = (xphi2 - yphi2)*rhophi2
      sphi1 = 2*xphi*yphi*rhophi2
      call mscat(lambda,chia2,xi,elke,beta2,qel,medium, spin_effects,fin
     *d_index,spin_index, w2,sint2)
5861  CONTINUE
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        xphi = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        xphi = 2*xphi - 1
        xphi2 = xphi*xphi
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        yphi = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        yphi2 = yphi*yphi
        rhophi2 = xphi2 + yphi2
        IF(rhophi2.LE.1)GO TO5862
      GO TO 5861
5862  CONTINUE
      rhophi2 = 1/rhophi2
      cphi2 = (xphi2 - yphi2)*rhophi2
      sphi2 = 2*xphi*yphi*rhophi2
      u2 = sint2*cphi2
      v2 = sint2*sphi2
      u2p = w1*u2 + sint1*w2
      us = u2p*cphi1 - v2*sphi1
      vs = u2p*sphi1 + v2*cphi1
      ws = w1*w2 - sint1*u2
      xi = 2*xi*xi_corr
      IF (( rng_seed .GT. 24 )) THEN
        call ranlux(rng_array)
        rng_seed = 1
      END IF
      eta = rng_array(rng_seed)
      rng_seed = rng_seed + 1
      eta = Sqrt(eta)
      eta1 = 0.5*(1 - eta)
      delta = 0.9082483-(0.1020621-0.0263747*gamma)*xi
      temp1 = 2 + tau
      temp = (2+tau*temp1)/((tau+1)*temp1)
      temp = temp - (tau+1)/((tau+2)*(chilog*(1+chia2)-1))
      temp = temp * epsilonp
      temp1 = 1 - temp
      delta = delta + 0.40824829*(epsilon*(tau+1)/((tau+2)* (chilog*(1+c
     *hia2)-1)*(chilog*(1+2*chia2)-2)) - 0.25*temp*temp)
      b = eta*delta
      c = eta*(1-delta)
      w1v2 = w1*v2
      ut = b*sint1*cphi1 + c*(cphi1*u2 - sphi1*w1v2) + eta1*us*temp1
      vt = b*sint1*sphi1 + c*(sphi1*u2 + cphi1*w1v2) + eta1*vs*temp1
      wt = eta1*(1+temp) + b*w1 + c*w2 + eta1*ws*temp1
      ustep = tustep*sqrt(ut*ut + vt*vt + wt*wt)
      sint02 = u0**2 + v0**2
      IF ((sint02 .GT. 1e-20)) THEN
        sint0 = sqrt(sint02)
        sint0i = 1/sint0
        cphi0 = sint0i*u0
        sphi0 = sint0i*v0
        u2p = w0*us + sint0*ws
        ws = w0*ws - sint0*us
        us = u2p*cphi0 - vs*sphi0
        vs = u2p*sphi0 + vs*cphi0
        u2p = w0*ut + sint0*wt
        wt = w0*wt - sint0*ut
        ut = u2p*cphi0 - vt*sphi0
        vt = u2p*sphi0 + vt*cphi0
      ELSE
        wt = w0*wt
        ws = w0*ws
      END IF
      xf = x0 + tustep*ut
      yf = y0 + tustep*vt
      zf = z0 + tustep*wt
      return
      end
      subroutine msdist_pI ( e0,eloss,tustep,rhof,medium,qel,spin_effect
     *s,u0,v0,w0,x0,y0,z0,  us,vs,ws,xf,yf,zf,ustep )
      implicit none
      real*8 e0,  eloss,  rhof,  tustep,  u0,  v0,  w0,  x0,  y0,  z0
      integer*4 medium, qel
      logical spin_effects
      real*8 us,  vs,  ws,  xf,  yf,  zf,  ustep
      real*8 blccc,  xcccc,  z,r,z2,r2,  r2max, chia2,  chilog,  cphi0,
     *  cphi,  sphi,  e,  elke,  beta2,  etap,  xi_corr,  ms_corr, epsil
     *on,  temp,  factor,  lambda,  p2,  p2i,  q1,  rhophi2,  sint,  sin
     *t0,  sint02,  sint0i,  sphi0,   u2p,  ut,  vt,  wt,  xi,  xphi,  x
     *phi2,  yphi,  yphi2
      logical find_index,  spin_index
      integer*4 lelke
      COMMON/ELECIN/ esig_e(1),psig_e(1), esige_max, psige_max, range_ep
     *(0:1,500,1), E_array(500,1), etae_ms0(500,1),etae_ms1(500,1),etap_
     *ms0(500,1),etap_ms1(500,1),q1ce_ms0(500,1),q1ce_ms1(500,1),q1cp_ms
     *0(500,1),q1cp_ms1(500,1),q2ce_ms0(500,1),q2ce_ms1(500,1),q2cp_ms0(
     *500,1),q2cp_ms1(500,1),blcce0(500,1),blcce1(500,1), EKE0(1),EKE1(1
     *), XR0(1),TEFF0(1),BLCC(1),XCC(1), ESIG0(500,1),ESIG1(500,1),PSIG0
     *(500,1),PSIG1(500,1),EDEDX0(500,1),EDEDX1(500,1),PDEDX0(500,1),PDE
     *DX1(500,1),EBR10(500,1),EBR11(500,1),PBR10(500,1),PBR11(500,1),PBR
     *20(500,1),PBR21(500,1),TMXS0(500,1),TMXS1(500,1), expeke1(1), IUNR
     *ST(1),EPSTFL(1),IAPRIM(1), sig_ismonotone(0:1,1)
      real*8 esig_e,   psig_e,   esige_max,  psige_max,  range_ep,  E_ar
     *ray,  etae_ms0,etae_ms1,  etap_ms0,etap_ms1,  q1ce_ms0,q1ce_ms1,
     *q1cp_ms0,q1cp_ms1,  q2ce_ms0,q2ce_ms1,  q2cp_ms0,q2cp_ms1,  blcce0
     *,blcce1,   expeke1,  EKE0,EKE1, XR0,  TEFF0,  BLCC,  XCC,  ESIG0,E
     *SIG1,  PSIG0,PSIG1,  EDEDX0,EDEDX1,  PDEDX0,PDEDX1,  EBR10,EBR11,
     * PBR10,PBR11,  PBR20,PBR21,  TMXS0,TMXS1
      integer*4 IUNRST,  EPSTFL,  IAPRIM
      logical sig_ismonotone
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/UPHIOT/THETA,SINTHE,COSTHE,SINPHI, COSPHI,PI,TWOPI,PI5D2
      real*8 THETA,  SINTHE,  COSTHE,  SINPHI,  COSPHI,  PI,TWOPI,PI5D2
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      common/emf_inputs/ExIN,EyIN,EzIN,  EMLMTIN,  BxIN, ByIN, BzIN,  Bx
     *, By, Bz,  Bx_new, By_new, Bz_new,  emfield_on
      real*8 ExIN,EyIN,EzIN, EMLMTIN, BxIN,ByIN,BzIN, Bx,By,Bz, Bx_new,B
     *y_new,Bz_new
      logical emfield_on
      blccc = blcc(medium)
      xcccc = xcc(medium)
      e = e0 - 0.5*eloss
      p2 = e*(e + rmt2)
      p2i = 1/p2
      chia2 = xcccc*p2i/(4*blccc)
      beta2 = p2/(p2 + rmsq)
      lambda = tustep*rhof*blccc/beta2
      factor = 1/(1 + 0.9784671*e)
      epsilon= eloss/e0
      epsilon= epsilon/(1-0.5*epsilon)
      temp = 0.25*(1 - factor*(1 - 0.333333*factor))*epsilon**2
      lambda = lambda*(1 + temp)
      IF (( spin_effects )) THEN
        elke = Log(e)
        Lelke=eke1(MEDIUM)*elke+eke0(MEDIUM)
        IF (( lelke .LT. 1 )) THEN
          lelke = 1
          elke = (1 - eke0(medium))/eke1(medium)
        END IF
        IF (( qel .EQ. 0 )) THEN
          etap=etae_ms1(Lelke,MEDIUM)*elke+etae_ms0(Lelke,MEDIUM)
          xi_corr=q1ce_ms1(Lelke,MEDIUM)*elke+q1ce_ms0(Lelke,MEDIUM)
        ELSE
          etap=etap_ms1(Lelke,MEDIUM)*elke+etap_ms0(Lelke,MEDIUM)
          xi_corr=q1cp_ms1(Lelke,MEDIUM)*elke+q1cp_ms0(Lelke,MEDIUM)
        END IF
        ms_corr=blcce1(Lelke,MEDIUM)*elke+blcce0(Lelke,MEDIUM)
      ELSE
        etap = 1
        xi_corr = 1
        ms_corr = 1
      END IF
      chia2 = xcccc*p2i/(4*blccc)*etap
      lambda = lambda/etap/(1+chia2)*ms_corr
      chilog = Log(1 + 1/chia2)
      q1 = 2*chia2*(chilog*(1 + chia2) - 1)
      xi = q1*lambda
      find_index = .true.
      spin_index = .true.
      call mscat(lambda,chia2,xi,elke,beta2,qel,medium, spin_effects,fin
     *d_index,spin_index, ws,sint)
5871  CONTINUE
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        xphi = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        xphi = 2*xphi - 1
        xphi2 = xphi*xphi
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        yphi = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        yphi2 = yphi*yphi
        rhophi2 = xphi2 + yphi2
        IF(rhophi2.LE.1)GO TO5872
      GO TO 5871
5872  CONTINUE
      rhophi2 = 1/rhophi2
      cphi = (xphi2 - yphi2)*rhophi2
      sphi = 2*xphi*yphi*rhophi2
      us = sint*cphi
      vs = sint*sphi
      xi = xi*xi_corr
      IF (( xi .LT. 0.1 )) THEN
        z = 1 - xi*(0.5 - xi*(0.166666667 - 0.041666667*xi))
      ELSE
        z = (1 - Exp(-xi))/xi
      END IF
      r = 0.5*sint
      r2 = r*r
      z2 = z*z
      r2max = 1 - z2
      IF (( r2max .LT. r2 )) THEN
        r2 = r2max
        r = Sqrt(r2)
      END IF
      ut = r*cphi
      vt = r*sphi
      wt = z
      ustep = Sqrt(z2 + r2)*tustep
      sint02 = u0**2 + v0**2
      IF ((sint02 .GT. 1e-20)) THEN
        sint0 = sqrt(sint02)
        sint0i = 1/sint0
        cphi0 = sint0i*u0
        sphi0 = sint0i*v0
        u2p = w0*us + sint0*ws
        ws = w0*ws - sint0*us
        us = u2p*cphi0 - vs*sphi0
        vs = u2p*sphi0 + vs*cphi0
        u2p = w0*ut + sint0*wt
        wt = w0*wt - sint0*ut
        ut = u2p*cphi0 - vt*sphi0
        vt = u2p*sphi0 + vt*cphi0
      ELSE
        wt = w0*wt
        ws = w0*ws
      END IF
      xf = x0 + tustep*ut
      yf = y0 + tustep*vt
      zf = z0 + tustep*wt
      return
      end
      SUBROUTINE PAIR
      implicit none
      COMMON/QDEBUG/QDEBUG
      LOGICAL QDEBUG
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      common/nrc_pair/ nrcp_fdata(65,84,1), nrcp_wdata(65,84,1), nrcp_id
     *ata(65,84,1), nrcp_xdata(65), nrcp_emin, nrcp_emax, nrcp_dle, nrcp
     *_dlei
      real*8 nrcp_fdata,nrcp_wdata,nrcp_xdata, nrcp_emin, nrcp_emax, nrc
     *p_dle, nrcp_dlei
      integer*4 nrcp_idata
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/UPHIOT/THETA,SINTHE,COSTHE,SINPHI, COSPHI,PI,TWOPI,PI5D2
      real*8 THETA,  SINTHE,  COSTHE,  SINPHI,  COSPHI,  PI,TWOPI,PI5D2
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      COMMON/EPCONT/EDEP,EDEP_LOCAL,TSTEP,TUSTEP,USTEP,TVSTEP,VSTEP, RHO
     *F,EOLD,ENEW,EKE,ELKE,GLE,E_RANGE, x_final,y_final,z_final, u_final
     *,v_final,w_final, IDISC,IROLD,IRNEW,IAUSFL(35)
      DOUBLE PRECISION EDEP,  EDEP_LOCAL
      real*8 TSTEP,  TUSTEP,  USTEP,  VSTEP,  TVSTEP,  RHOF,  EOLD,  ENE
     *W,  EKE,  ELKE,  GLE,  E_RANGE, x_final,y_final,z_final,  u_final,
     *v_final,w_final
      integer*4 IDISC,  IROLD,  IRNEW,  IAUSFL
      common/triplet_data/ a_triplet(250,1), b_triplet(250,1), dl_triple
     *t, dli_triplet, bli_triplet, log_4rm
      real*8 a_triplet,b_triplet,dl_triplet, dli_triplet, bli_triplet, l
     *og_4rm
      common/egs_vr/ e_max_rr(1),  prob_RR,  nbr_split,  i_play_RR,
     * i_survived_RR,
     *     n_RR_warning,                                        i_do_rr(
     *1)
      real*8          e_max_rr,prob_RR
      integer*4       nbr_split,i_play_RR,i_survived_RR,n_RR_warning
      integer*2     i_do_rr
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      DOUBLE PRECISION PEIG,  PESE1,  PESE2
      real*8 EIG,  ESE2,  RNNO30,RNNO31,rnno32,rnno33,rnno34,  DELTA,  R
     *EJF,  rejmax,  aux1,aux2,  Amax,  Bmax,  del0,  br,
     *                               Eminus,Eplus,Eavail,rnno_RR
      integer*4
     *                     L,L1
      real*8 ESE,  PSE,  ZTARG,  TTEIG,  TTESE,  TTPSE,  ESEDEI, ESEDER,
     * XIMIN,  XIMID,  REJMIN, REJMID, REJTOP, YA,XITRY,GALPHA,GBETA,  X
     *ITST,  REJTST_on_REJTOP ,  REJTST, RTEST
      integer*4 ICHRG
      real*8 k,xx,abin,rbin,alias_sample1
      integer*4 ibin, iq1, iq2, iprdst_use
      logical do_nrc_pair
      integer*4 itrip
      real*8 ftrip
      NPold = NP
      IF (( i_play_RR .EQ. 1 )) THEN
        i_survived_RR = 0
        IF (( prob_RR .LE. 0 )) THEN
          IF (( n_RR_warning .LT. 50 )) THEN
            n_RR_warning = n_RR_warning + 1
            write(i_log,'(/a)') '***************** Warning: '
            write(i_log,'(a,g14.6)') 'Attempt to play Russian Roulette w
     *ith prob_RR<0! '
          END IF
        ELSE
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          rnno_RR = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          IF (( rnno_RR .GT. prob_RR )) THEN
            i_survived_RR =2
            IF (( np .GT. 1 )) THEN
              np = np-1
            ELSE
              wt(np) = 0
              e(np) = 0
            END IF
            return
          ELSE
            wt(np) = wt(np)/prob_RR
          END IF
        END IF
      END IF
      IF (( np+1 .GT. 2 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,'(//,3a,/,2(a,i9))') ' In subroutine ','PAIR', ' sta
     *ck size exceeded! ',' $MAXSTACK = ',2,' np = ',np+1
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      PEIG=E(NP)
      EIG=PEIG
      do_nrc_pair = .false.
      IF (( itriplet .GT. 0 .AND. eig .GT. 4*rm )) THEN
        itrip = dli_triplet*gle + bli_triplet
        ftrip = a_triplet(itrip,medium)*gle + b_triplet(itrip,medium)
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        rnno34 = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        IF (( rnno34 .LT. ftrip )) THEN
          call sample_triplet
          return
        END IF
      END IF
      IF (( pair_nrc .EQ. 1 )) THEN
        k = eig/rm
        IF (( k .LT. nrcp_emax )) THEN
          do_nrc_pair = .true.
          IF (( k .LE. nrcp_emin )) THEN
            ibin = 1
          ELSE
            abin = 1 + log((k-2)/(nrcp_emin-2))*nrcp_dlei
            ibin = abin
            abin = abin - ibin
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            rbin = rng_array(rng_seed)
            rng_seed = rng_seed + 1
            IF((rbin .LT. abin))ibin = ibin + 1
          END IF
          xx = alias_sample1(64,nrcp_xdata, nrcp_fdata(1,ibin,medium),nr
     *    cp_wdata(1,ibin,medium), nrcp_idata(1,ibin,medium))
          IF (( xx .GT. 0.5 )) THEN
            pese1 = prm*(1 + xx*(k-2))
            iq1 = 1
            pese2 = peig - pese1
            iq2 = -1
          ELSE
            pese2 = prm*(1 + xx*(k-2))
            iq2 = 1
            pese1 = peig - pese2
            iq1 = -1
          END IF
        END IF
      END IF
      IF (( .NOT.do_nrc_pair )) THEN
        IF ((EIG.LE.2.1)) THEN
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          RNNO30 = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          rnno34 = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          PESE2 = PRM + 0.5*RNNO30*(PEIG-2*PRM)
          PESE1 = PEIG - PESE2
          IF (( rnno34 .LT. 0.5 )) THEN
            iq1 = -1
            iq2 = 1
          ELSE
            iq1 = 1
            iq2 = -1
          END IF
        ELSE
          IF ((EIG.LT.50.)) THEN
            L = 5
            L1 = L + 1
            delta = 4*delcm(medium)/eig
            IF (( delta .LT. 1 )) THEN
              Amax = dl1(l,medium)+delta*(dl2(l,medium)+delta*dl3(l,medi
     *        um))
              Bmax = dl1(l1,medium)+delta*(dl2(l1,medium)+delta*dl3(l1,m
     *        edium))
            ELSE
              aux2 = log(delta+dl6(l,medium))
              Amax = dl4(l,medium)+dl5(l,medium)*aux2
              Bmax = dl4(l1,medium)+dl5(l1,medium)*aux2
            END IF
            aux1 = 1 - rmt2/eig
            aux1 = aux1*aux1
            aux1 = aux1*Amax/3
            aux1 = aux1/(Bmax+aux1)
          ELSE
            L = 7
            Amax = dl1(l,medium)
            Bmax = dl1(l+1,medium)
            aux1 = bpar(2,medium)*(1-bpar(1,medium)*rm/eig)
          END IF
          del0 = eig*delcm(medium)
          Eavail = eig - rmt2
5881      CONTINUE
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            RNNO30 = rng_array(rng_seed)
            rng_seed = rng_seed + 1
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            RNNO31 = rng_array(rng_seed)
            rng_seed = rng_seed + 1
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            RNNO34 = rng_array(rng_seed)
            rng_seed = rng_seed + 1
            IF (( rnno30 .GT. aux1 )) THEN
              br = 0.5*rnno31
              rejmax = Bmax
              l1 = l+1
            ELSE
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              rnno32 = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              rnno33 = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              br = 0.5*(1-max(rnno31,rnno32,rnno33))
              rejmax = Amax
              l1 = l
            END IF
            Eminus = br*Eavail + rm
            Eplus = eig - Eminus
            delta = del0/(Eminus*Eplus)
            IF (( delta .LT. 1 )) THEN
              rejf = dl1(l1,medium)+delta*(dl2(l1,medium)+delta*dl3(l1,m
     *        edium))
            ELSE
              rejf = dl4(l1,medium)+dl5(l1,medium)*log(delta+dl6(l1,medi
     *        um))
            END IF
            IF((( rnno34*rejmax .LE. rejf )))GO TO5882
          GO TO 5881
5882      CONTINUE
          pese2 = Eminus
          pese1 = peig - pese2
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          RNNO34 = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          IF (( rnno34 .LT. 0.5 )) THEN
            iq1 = -1
            iq2 = 1
          ELSE
            iq1 = 1
            iq2 = -1
          END IF
        END IF
      END IF
      ESE2=PESE2
      E(NP)=PESE1
      E(NP+1)=PESE2
      IF (( iprdst .GT. 0 )) THEN
        IF (( iprdst .EQ. 4 )) THEN
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          rtest = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          gbeta = PESE1/(PESE1+10)
          IF (( rtest .LT. gbeta )) THEN
            iprdst_use = 1
          ELSE
            iprdst_use = 4
          END IF
        ELSE IF(( iprdst .EQ. 2 .AND. eig .LT. 4.14 )) THEN
          iprdst_use = 1
        ELSE
          iprdst_use = iprdst
        END IF
        DO 5891 ichrg=1,2
          IF ((ICHRG.EQ.1)) THEN
            ESE=PESE1
          ELSE
            ESE=ESE2
            IF (( iprdst .EQ. 4 )) THEN
              gbeta = ESE/(ESE+10)
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              rtest = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              IF (( rtest .LT. gbeta )) THEN
                iprdst_use = 1
              ELSE
                iprdst_use = 4
              END IF
            END IF
          END IF
          IF (( iprdst_use .EQ. 1 )) THEN
            PSE=SQRT(MAX(0.0,(ESE-RM)*(ESE+RM)))
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            COSTHE = rng_array(rng_seed)
            rng_seed = rng_seed + 1
            COSTHE=1.0-2.0*COSTHE
            SINTHE=RM*SQRT((1.0-COSTHE)*(1.0+COSTHE))/(PSE*COSTHE+ESE)
            COSTHE=(ESE*COSTHE+PSE)/(PSE*COSTHE+ESE)
          ELSE IF(( iprdst_use .EQ. 2 )) THEN
            ZTARG=ZBRANG(MEDIUM)
            TTEIG=EIG/RM
            TTESE=ESE/RM
            TTPSE=SQRT((TTESE-1.0)*(TTESE+1.0))
            ESEDEI=TTESE/(TTEIG-TTESE)
            ESEDER=1.0/ESEDEI
            XIMIN=1.0/(1.0+(3.141593*TTESE)**2)
            REJMIN = 2.0+3.0*(ESEDEI+ESEDER) - 4.00*(ESEDEI+ESEDER+1.0-4
     *      .0*(XIMIN-0.5)**2)*( 1.0+0.25*LOG( ((1.0+ESEDER)*(1.0+ESEDEI
     *      )/(2.*TTEIG))**2+ZTARG*XIMIN**2 ) )
            YA=(2.0/TTEIG)**2
            XITRY=MAX(0.01,MAX(XIMIN,MIN(0.5,SQRT(YA/ZTARG))))
            GALPHA=1.0+0.25*LOG(YA+ZTARG*XITRY**2)
            GBETA=0.5*ZTARG*XITRY/(YA+ZTARG*XITRY**2)
            GALPHA=GALPHA-GBETA*(XITRY-0.5)
            XIMID=GALPHA/(3.0*GBETA)
            IF ((GALPHA.GE.0.0)) THEN
              XIMID=0.5-XIMID+SQRT(XIMID**2+0.25)
            ELSE
              XIMID=0.5-XIMID-SQRT(XIMID**2+0.25)
            END IF
            XIMID=MAX(0.01,MAX(XIMIN,MIN(0.5,XIMID)))
            REJMID = 2.0+3.0*(ESEDEI+ESEDER) - 4.00*(ESEDEI+ESEDER+1.0-4
     *      .0*(XIMID-0.5)**2)*( 1.0+0.25*LOG( ((1.0+ESEDER)*(1.0+ESEDEI
     *      )/(2.*TTEIG))**2+ZTARG*XIMID**2 ) )
            REJTOP=1.02*MAX(REJMIN,REJMID)
5901        CONTINUE
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              XITST = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              REJTST = 2.0+3.0*(ESEDEI+ESEDER) - 4.00*(ESEDEI+ESEDER+1.0
     *        -4.0*(XITST-0.5)**2)*( 1.0+0.25*LOG( ((1.0+ESEDER)*(1.0+ES
     *        EDEI)/(2.*TTEIG))**2+ZTARG*XITST**2 ) )
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              RTEST = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              THETA=SQRT(1.0/XITST-1.0)/TTESE
              REJTST_on_REJTOP = REJTST/REJTOP
              IF((((RTEST .LE. REJTST_on_REJTOP) .AND. (THETA .LT. PI) )
     *        ))GO TO5902
            GO TO 5901
5902        CONTINUE
            SINTHE=SIN(THETA)
            COSTHE=COS(THETA)
          ELSE IF(( iprdst_use .EQ. 3 )) THEN
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            COSTHE = rng_array(rng_seed)
            rng_seed = rng_seed + 1
            COSTHE=1.0-2.0*COSTHE
            sinthe=(1-costhe)*(1+costhe)
            IF (( sinthe .GT. 0 )) THEN
              sinthe = sqrt(sinthe)
            ELSE
              sinthe = 0
            END IF
          ELSE
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            costhe = rng_array(rng_seed)
            rng_seed = rng_seed + 1
            costhe=1-2*sqrt(costhe)
            sinthe=(1-costhe)*(1+costhe)
            IF (( sinthe .GT. 0 )) THEN
              sinthe=sqrt(sinthe)
            ELSE
              sinthe=0
            END IF
          END IF
          IF (( ichrg .EQ. 1 )) THEN
            CALL UPHI(2,1)
          ELSE
            sinthe=-sinthe
            NP=NP+1
            CALL UPHI(3,2)
          END IF
5891    CONTINUE
5892    CONTINUE
        iq(np) = iq2
        iq(np-1) = iq1
        return
      ELSE
        THETA=0
      END IF
      CALL UPHI(1,1)
      NP=NP+1
      SINTHE=-SINTHE
      CALL UPHI(3,2)
      IQ(NP)=iq2
      IQ(NP-1)=iq1
      RETURN
      END
      subroutine sample_triplet
      implicit none
      COMMON/EPCONT/EDEP,EDEP_LOCAL,TSTEP,TUSTEP,USTEP,TVSTEP,VSTEP, RHO
     *F,EOLD,ENEW,EKE,ELKE,GLE,E_RANGE, x_final,y_final,z_final, u_final
     *,v_final,w_final, IDISC,IROLD,IRNEW,IAUSFL(35)
      DOUBLE PRECISION EDEP,  EDEP_LOCAL
      real*8 TSTEP,  TUSTEP,  USTEP,  VSTEP,  TVSTEP,  RHOF,  EOLD,  ENE
     *W,  EKE,  ELKE,  GLE,  E_RANGE, x_final,y_final,z_final,  u_final,
     *v_final,w_final
      integer*4 IDISC,  IROLD,  IRNEW,  IAUSFL
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      real*8 fmax_array(250), eta_p_array(250), eta_Ep_array(250), eta_c
     *ostp_array(250), eta_costm_array(250), ebin_array(250), wp_array(2
     *50), qmin_array(250)
      real*8 kmin, kmax, dlogki, alogkm, prmi, tiny_eta
      real*8 ai,rnno,k,qmin,qmax,aux,a1,a2,a3,D,px1,px2,pp_min,pp_max, E
     *p_min,Ep_max,k2p2,k2p2x,peig,b,aux1,aux12,D1,aux3,xmin,xmax, aux6,
     *aux7,uu,cphi,sphi,cphi_factor,aux5,phi,tmp
      real*8 Er,pr,pr2,eta_pr
      real*8 Ep,pp,pp2,wEp,cost_p,sint_p,eta_Ep,mup_min,wmup, eta_costp,
     *Epp,pp_sintp,pp_sntp2
      real*8 Em,pm,pm2,cost_m,sint_m,Emm,wmum,pm_sintm, eta_costm
      real*8 k2,k3,s2,s3,k2k3i,k22,k32,q2,aux4,S_1,S_2,sigma
      real*8 ppx, ppy, ppz, pmx, pmy, pmz, prx, pry, prz, a,c,sindel,cos
     *del,sinpsi
      integer*4 i
      logical use_it
      integer*4 iscore
      logical is_initialized
      data is_initialized/.false./
      save is_initialized,fmax_array,eta_p_array,eta_Ep_array,eta_costp_
     *array, eta_costm_array,ebin_array,wp_array,qmin_array, kmin,kmax,d
     *logki,alogkm,prmi,tiny_eta
      IF (( .NOT.is_initialized )) THEN
        is_initialized = .true.
        tiny_eta = 1e-6
        DO 5911 i=1,250
          fmax_array(i) = -1
5911    CONTINUE
5912    CONTINUE
        kmax = 0
        kmin = 4.1*prm
        DO 5921 i=1,nmed
          IF((up(i) .GT. kmax))kmax = UP(i)
5921    CONTINUE
5922    CONTINUE
        IF((kmax .LE. kmin))return
        dlogki = 250 - 1
        dlogki = dlogki/log(kmax/kmin)
        alogkm = 1 - dlogki*log(kmin)
        prmi = 1/prm
        DO 5931 i=1,250
          k = 4.1*exp((i-1.)/dlogki)
          ebin_array(i) = k
          qmin = 4*k/(k*(k-1)+(k+1)*sqrt(k*(k-4)))
          qmax = (k*(k-1) + (k+1)*sqrt(k*(k-4)))/(2*k+1)
          qmin_array(i) = qmin
          wp_array(i) = log(qmax/qmin)
5931    CONTINUE
5932    CONTINUE
      END IF
      peig = e(np)
      IF((peig .LE. 4*prm))return
      IF (( np+2 .GT. 2 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,'(//,3a,/,2(a,i9))') ' In subroutine ','sample_tripl
     *et', ' stack size exceeded! ',' $MAXSTACK = ',2,' np = ',np+2
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      IF (( peig .LE. kmin )) THEN
        i = 1
      ELSE IF(( peig .GE. kmax )) THEN
        i = 250
      ELSE
        ai = alogkm + dlogki*gle
        i = ai
        ai = ai - i
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        rnno = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        IF (( rnno .LT. ai )) THEN
          i = i+1
        END IF
      END IF
      k = ebin_array(i)
5940  CONTINUE
      IF (( rng_seed .GT. 24 )) THEN
        call ranlux(rng_array)
        rng_seed = 1
      END IF
      eta_pr = rng_array(rng_seed)
      rng_seed = rng_seed + 1
      IF((eta_pr .LT. tiny_eta))eta_pr = tiny_eta
      pr = qmin_array(i)*exp(eta_pr*wp_array(i))
      pr2 = pr*pr
      Er = sqrt(1+pr2)
      aux = Er-pr-1
      a1=(k-pr)*(1-Er-k*aux)
      a2=1+k-Er
      a3=1/(aux*(pr+Er-2*k-1))
      D = a2*sqrt(aux*(2*k*Er+k*k*aux-pr*(Er+pr+1)/2))
      px1 = (a1 + D)*a3
      px2 = (a1 - D)*a3
      IF (( px1 .LT. px2 )) THEN
        pp_min = px1
        pp_max = px2
      ELSE
        pp_min = px2
        pp_max = px1
      END IF
      Ep_min = sqrt(1 + pp_min*pp_min)
      Ep_max = sqrt(1 + pp_max*pp_max)
      IF (( rng_seed .GT. 24 )) THEN
        call ranlux(rng_array)
        rng_seed = 1
      END IF
      eta_Ep = rng_array(rng_seed)
      rng_seed = rng_seed + 1
      IF((eta_Ep .LT. tiny_eta))eta_Ep = tiny_eta
      wEp = Ep_max - Ep_min
      Ep = Ep_min + eta_Ep*wEp
      pp2 = Ep*Ep - 1
      pp = sqrt(pp2)
      k2p2 = k*k + pp2
      Em = k + 1 - Er - Ep
      pm2 = Em*Em-1
      pm = sqrt(pm2)
      mup_min = (k2p2 - (pr + pm)*(pr + pm))/(2*k*pp)
      IF (( rng_seed .GT. 24 )) THEN
        call ranlux(rng_array)
        rng_seed = 1
      END IF
      eta_costp = rng_array(rng_seed)
      rng_seed = rng_seed + 1
      IF((eta_costp .LT. tiny_eta))eta_costp = tiny_eta
      Epp = Ep/pp
      wmup = log((Epp-1)/(Epp-mup_min))
      cost_p = Epp - (Epp - mup_min)*exp(wmup*eta_costp)
      wmup = wmup*(cost_p - Epp)
      sint_p = 1-cost_p*cost_p
      IF (( sint_p .GT. 1e-20 )) THEN
        sint_p = sqrt(sint_p)
      ELSE
        sint_p = 1e-10
      END IF
      k2p2x = k2p2 - 2*k*pp*cost_p
      b = pr2-k2p2x-pm2
      aux1 = k - pp*cost_p
      aux12 = aux1*aux1
      pp_sintp = pp*sint_p
      pp_sntp2 = pp_sintp*pp_sintp
      D1 = pm2*(aux12+pp_sntp2)-b*b/4
      IF (( D1 .LE. 0 )) THEN
        goto 5940
      END IF
      D = 2*pp_sintp*sqrt(D1)
      aux3 = 0.5/(aux12+pp_sntp2)
      xmin = (-b*aux1-D)*aux3
      xmax = (-b*aux1+D)*aux3
      IF (( rng_seed .GT. 24 )) THEN
        call ranlux(rng_array)
        rng_seed = 1
      END IF
      eta_costm = rng_array(rng_seed)
      rng_seed = rng_seed + 1
      IF((eta_costm .LT. tiny_eta))eta_costm = tiny_eta
      aux6 = sqrt((Em-xmin)/(Em-xmax))
      aux7 = aux6*tan(1.570796326794897*eta_costm)
      uu = (aux7-1)/(aux7+1)
      cost_m = 0.5*(xmax + xmin + 2*uu*(xmax-xmin)/(1+uu*uu))
      wmum = sqrt((xmax-cost_m)*(cost_m-xmin))
      wmum = wmum*aux6*(Em-cost_m)/(Em-xmin)
      cost_m = cost_m/pm
      sint_m = sqrt(1-cost_m*cost_m)
      pm_sintm = pm*sint_m
      cphi = (b + 2*pm*cost_m*aux1)/(2*pp_sintp*pm_sintm)
      IF (( abs(cphi) .GE. 1 )) THEN
        goto 5940
      END IF
      sphi = sqrt(1-cphi*cphi)
      k3 = k*(pp*cost_p - Ep)
      k2 = k*(pm*cost_m - Em)
      k22 = k2*k2
      k32 = k3*k3
      k2k3i = 1/(k2*k3)
      s2 = pp*pm*(cost_p*cost_m + sint_p*sint_m*cphi) - Ep*Em
      s3 = k2 - Em + 1 - s2
      q2 = 2*(Er-1)
      S_1 = k32+k22+(q2-2)*s2-(1-q2/2)*(k32+k22)*k2k3i
      aux4 = k3*Ep-k2*Em
      S_2 = -q2*(Ep*Ep+Em*Em) + 2*s2 - (2*aux4*aux4 - k22 - k32)*k2k3i
      sigma = abs(pp*pm2*pm*k2k3i/(q2*q2*(Em*s3+Er))*(S_1*(1-q2/4)+S_2*(
     *1+q2/4)))
      cphi_factor = abs(2*Er*pm2-Em*(k2p2x-pr2-pm2))/(2*pp_sintp*pm_sint
     *m*pm2*sphi)
      sigma = sigma*cphi_factor*wEp*wmup*wmum*wp_array(i)*pr2/Er
      IF (( sigma .LT. 0 )) THEN
        write(i_log,'(/a)') '***************** Warning: '
        write(i_log,*) 'In triplet sigma < 0 ? ',sigma
      END IF
      use_it = .true.
      IF (( sigma .LT. fmax_array(i) )) THEN
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        rnno = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        IF (( sigma .LT. fmax_array(i)*rnno )) THEN
          use_it = .false.
        END IF
      END IF
      IF (( use_it )) THEN
        fmax_array(i) = sigma
        eta_p_array(i) = eta_pr
        eta_Ep_array(i) = eta_Ep
        eta_costp_array(i) = eta_costp
        eta_costm_array(i) = eta_costm
      ELSE
        eta_pr = eta_p_array(i)
        eta_Ep = eta_Ep_array(i)
        eta_costp = eta_costp_array(i)
        eta_costm = eta_costm_array(i)
      END IF
      k = peig*prmi
      aux5 = k*(k-1)+(k+1)*sqrt(k*(k-4))
      qmin = 4*k/aux5
      qmax = aux5/(2*k+1)
      pr = qmin*exp(eta_pr*log(qmax/qmin))
      pr2 = pr*pr
      Er = sqrt(1+pr2)
      aux = Er-pr-1
      a1=(k-pr)*(1-Er-k*aux)
      a2=1+k-Er
      a3=1/(aux*(pr+Er-2*k-1))
      D = a2*sqrt(aux*(2*k*Er+k*k*aux-pr*(Er+pr+1)/2))
      px1 = (a1 + D)*a3
      px2 = (a1 - D)*a3
      IF (( px1 .LT. px2 )) THEN
        pp_min = px1
        pp_max = px2
      ELSE
        pp_min = px2
        pp_max = px1
      END IF
      Ep_min = sqrt(1 + pp_min*pp_min)
      Ep_max = sqrt(1 + pp_max*pp_max)
      wEp = Ep_max - Ep_min
      Ep = Ep_min + eta_Ep*wEp
      pp2 = Ep*Ep - 1
      pp = sqrt(pp2)
      k2p2 = k*k + pp2
      Em = k + 1 - Er - Ep
      pm2 = Em*Em-1
      pm = sqrt(pm2)
      mup_min = (k2p2 - (pr + pm)*(pr + pm))/(2*k*pp)
      Epp = Ep/pp
      wmup = log((Epp-1)/(Epp-mup_min))
      cost_p = Epp - (Epp - mup_min)*exp(wmup*eta_costp)
      sint_p = sqrt(1-cost_p*cost_p)
      k2p2x = k2p2 - 2*k*pp*cost_p
      b = pr2-k2p2x-pm2
      aux1 = k - pp*cost_p
      aux12 = aux1*aux1
      pp_sintp = pp*sint_p
      pp_sntp2 = pp_sintp*pp_sintp
      D1 = pm2*(aux12+pp_sntp2)-b*b/4
      IF (( D1 .LE. 0 )) THEN
        goto 5940
      END IF
      D = 2*pp_sintp*sqrt(D1)
      aux3 = 0.5/(aux12+pp_sntp2)
      xmin = (-b*aux1-D)*aux3
      xmax = (-b*aux1+D)*aux3
      aux6 = sqrt((Em-xmin)/(Em-xmax))
      aux7 = aux6*tan(1.570796326794897*eta_costm)
      uu = (aux7-1)/(aux7+1)
      cost_m = 0.5*(xmax + xmin + 2*uu*(xmax-xmin)/(1+uu*uu))/pm
      sint_m = sqrt(1-cost_m*cost_m)
      pm_sintm = pm*sint_m
      cphi = (b + 2*pm*cost_m*aux1)/(2*pp_sintp*pm_sintm)
      IF (( abs(cphi) .GE. 1 )) THEN
        goto 5940
      END IF
      sphi = sqrt(1-cphi*cphi)
      IF (( rng_seed .GT. 24 )) THEN
        call ranlux(rng_array)
        rng_seed = 1
      END IF
      phi = rng_array(rng_seed)
      rng_seed = rng_seed + 1
      phi = phi*6.283185307179586
      ppx = pp*sint_p
      ppy = 0
      pmx = pm*sint_m*cphi
      pmy = pm*sint_m*sphi
      cphi = cos(phi)
      sphi = sin(phi)
      tmp = ppx*sphi
      ppx = ppx*cphi - ppy*sphi
      ppy = tmp + ppy*cphi
      tmp = pmx*sphi
      pmx = pmx*cphi - pmy*sphi
      pmy = tmp + pmy*cphi
      ppz = pp*cost_p
      pmz = pm*cost_m
      prx = -ppx-pmx
      pry = -ppy-pmy
      prz = k - ppz - pmz
      NPold = np
      X(np)=X(np)
      Y(np)=Y(np)
      Z(np)=Z(np)
      IR(np)=IR(np)
      WT(np)=WT(np)
      DNEAR(np)=DNEAR(np)
      LATCH(np)=LATCH(np)
      X(np+1)=X(np)
      Y(np+1)=Y(np)
      Z(np+1)=Z(np)
      IR(np+1)=IR(np)
      WT(np+1)=WT(np)
      DNEAR(np+1)=DNEAR(np)
      LATCH(np+1)=LATCH(np)
      X(np+2)=X(np+1)
      Y(np+2)=Y(np+1)
      Z(np+2)=Z(np+1)
      IR(np+2)=IR(np+1)
      WT(np+2)=WT(np+1)
      DNEAR(np+2)=DNEAR(np+1)
      LATCH(np+2)=LATCH(np+1)
      pp = 1/pp
      pm = 1/pm
      pr = 1/pr
      a = u(np)
      b = v(np)
      c = w(np)
      sinpsi = a*a + b*b
      IF (( sinpsi .GT. 1e-20 )) THEN
        sinpsi = sqrt(sinpsi)
        sindel = b/sinpsi
        cosdel = a/sinpsi
        IF (( Ep .GT. Em )) THEN
          u(np) = pp*(c*cosdel*ppx - sindel*ppy + a*ppz)
          v(np) = pp*(c*sindel*ppx + cosdel*ppy + b*ppz)
          w(np) = pp*(c*ppz - sinpsi*ppx)
          iq(np) = 1
          E(np) = Ep*prm
          u(np+1) = pm*(c*cosdel*pmx - sindel*pmy + a*pmz)
          v(np+1) = pm*(c*sindel*pmx + cosdel*pmy + b*pmz)
          w(np+1) = pm*(c*pmz - sinpsi*pmx)
          iq(np+1) = -1
          E(np+1) = Em*prm
        ELSE
          u(np+1) = pp*(c*cosdel*ppx - sindel*ppy + a*ppz)
          v(np+1) = pp*(c*sindel*ppx + cosdel*ppy + b*ppz)
          w(np+1) = pp*(c*ppz - sinpsi*ppx)
          iq(np+1) = 1
          E(np+1) = Ep*prm
          u(np) = pm*(c*cosdel*pmx - sindel*pmy + a*pmz)
          v(np) = pm*(c*sindel*pmx + cosdel*pmy + b*pmz)
          w(np) = pm*(c*pmz - sinpsi*pmx)
          iq(np) = -1
          E(np) = Em*prm
        END IF
        np = np + 2
        u(np) = pr*(c*cosdel*prx - sindel*pry + a*prz)
        v(np) = pr*(c*sindel*prx + cosdel*pry + b*prz)
        w(np) = pr*(c*prz - sinpsi*prx)
        iq(np) = -1
        E(np) = Er*prm
      ELSE
        IF (( Ep .GT. Em )) THEN
          u(np) = pp*ppx
          v(np) = pp*ppy
          w(np) = c*pp*ppz
          iq(np) = 1
          E(np) = Ep*prm
          u(np+1) = pm*pmx
          v(np+1) = pm*pmy
          w(np+1) = c*pm*pmz
          iq(np+1) = -1
          E(np+1) = Em*prm
        ELSE
          u(np+1) = pp*ppx
          v(np+1) = pp*ppy
          w(np+1) = c*pp*ppz
          iq(np+1) = 1
          E(np+1) = Ep*prm
          u(np) = pm*pmx
          v(np) = pm*pmy
          w(np) = c*pm*pmz
          iq(np) = -1
          E(np) = Em*prm
        END IF
        np = np + 2
        u(np) = pr*prx
        v(np) = pr*pry
        w(np) = c*pr*prz
        iq(np) = -1
        E(np) = Er*prm
      END IF
      return
      end
      SUBROUTINE PHOTO
      implicit none
      COMMON/BOUNDS/ECUT(1),PCUT(1),VACDST
      real*8 ECUT,  PCUT,  VACDST
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      COMMON/QDEBUG/QDEBUG
      LOGICAL QDEBUG
      COMMON/EDGE/binding_energies(30,100), interaction_prob(6,100), rel
     *axation_prob(39,100), edge_energies(16,100), edge_number(100), edg
     *e_a(16,100), edge_b(16,100), edge_c(16,100), edge_d(16,100), IEDGF
     *L(1),IPHTER(1)
      real*8 binding_energies,  interaction_prob,    relaxation_prob,  e
     *dge_energies,  edge_a,edge_b,edge_c,edge_d
      integer*2 IEDGFL,  IPHTER
      integer*4 edge_number
      COMMON/EPCONT/EDEP,EDEP_LOCAL,TSTEP,TUSTEP,USTEP,TVSTEP,VSTEP, RHO
     *F,EOLD,ENEW,EKE,ELKE,GLE,E_RANGE, x_final,y_final,z_final, u_final
     *,v_final,w_final, IDISC,IROLD,IRNEW,IAUSFL(35)
      DOUBLE PRECISION EDEP,  EDEP_LOCAL
      real*8 TSTEP,  TUSTEP,  USTEP,  VSTEP,  TVSTEP,  RHOF,  EOLD,  ENE
     *W,  EKE,  ELKE,  GLE,  E_RANGE, x_final,y_final,z_final,  u_final,
     *v_final,w_final
      integer*4 IDISC,  IROLD,  IRNEW,  IAUSFL
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/PHOTIN/ EBINDA(1), GE0(1),GE1(1), GMFP0(2000,1),GMFP1(2000,
     *1),GBR10(2000,1),GBR11(2000,1),GBR20(2000,1),GBR21(2000,1), RCO0(1
     *),RCO1(1), RSCT0(100,1),RSCT1(100,1), COHE0(2000,1),COHE1(2000,1),
     *  PHOTONUC0(2000,1),PHOTONUC1(2000,1), DPMFP, MPGEM(1,1), NGR(1)
      real*8 EBINDA,  GE0,GE1,  GMFP0,GMFP1,  GBR10,GBR11,  GBR20,GBR21,
     *  RCO0,RCO1,  RSCT0,RSCT1,  COHE0,COHE1,   PHOTONUC0,PHOTONUC1,  D
     *PMFP
      integer*4
     *                  MPGEM,
     *                          NGR
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/UPHIOT/THETA,SINTHE,COSTHE,SINPHI, COSPHI,PI,TWOPI,PI5D2
      real*8 THETA,  SINTHE,  COSTHE,  SINPHI,  COSPHI,  PI,TWOPI,PI5D2
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      common/x_options/eadl_relax,  mcdf_pe_xsections
      logical eadl_relax, mcdf_pe_xsections
      common/egs_vr/ e_max_rr(1),  prob_RR,  nbr_split,  i_play_RR,
     * i_survived_RR,
     *     n_RR_warning,                                        i_do_rr(
     *1)
      real*8          e_max_rr,prob_RR
      integer*4       nbr_split,i_play_RR,i_survived_RR,n_RR_warning
      integer*2     i_do_rr
      common/relax_data/ relax_first(3000),  relax_ntran(3000),  relax_s
     *tate(10000),  relax_prob(10000),  relax_atbin(10000),  relax_ntot
      real*8 relax_prob
      integer*4 relax_first, relax_ntran, relax_state, relax_atbin, rela
     *x_ntot
      real*8 EELEC,  BETA,  GAMMA,  ALPHA,  RATIO,  RNPHT,  FKAPPA, XI,
     * SINTH2, RNPHT2
      DOUBLE PRECISION PEIG
      real*8 BR,  sigma,  aux,aux1,  probs(50),  sigtot,  e_vac,  rnno_R
     *R
      integer*4 IARG,  iZ,   irl,  ints(50),  j,ip,  n_warning,  k
      logical do_relax
      save n_warning
      data n_warning/0/
      IF (( mcdf_pe_xsections )) THEN
        call egs_shellwise_photo()
        return
      END IF
      NPold = NP
      PEIG=E(NP)
      irl = ir(np)
      IF (( peig .LT. edge_energies(2,1) )) THEN
        IF (( n_warning .LT. 100 )) THEN
          n_warning = n_warning + 1
          write(i_log,*) ' Subroutine PHOTO called with E = ',peig, ' wh
     *ich is below the current min. energy of 1 keV! '
          write(i_log,*) ' Converting now this photon to an electron, '
          write(i_log,*) ' but you should check your code! '
        END IF
        iq(np) = -1
        e(np) = peig + prm
        return
      END IF
      iZ = iedgfl(irl)
      do_relax = .false.
      edep = pzero
      IF (( iedgfl(irl) .NE. 0 )) THEN
        IF (( nne(medium) .EQ. 1 )) THEN
          iZ = int( zelem(medium,1) + 0.5 )
          DO 5951 j=1,edge_number(iZ)
            IF((peig .GE. edge_energies(j,iZ)))GO TO5952
5951      CONTINUE
5952      CONTINUE
        ELSE
          aux = peig*peig
          aux1 = aux*peig
          aux = aux*Sqrt(peig)
          sigtot = 0
          DO 5961 k=1,nne(medium)
            iZ = int( zelem(medium,k) + 0.5 )
            IF (( iZ .LT. 1 .OR. iZ .GT. 100 )) THEN
              write(i_log,*) ' Error in PHOTO: '
              write(i_log,'(/a)') '***************** Error: '
              write(i_log,*) '   Atomic number of element ',k, ' in medi
     *um ',medium,' is not between 1 and ',100
              write(i_log,'(/a)') '***************** Quiting now.'
              call exit(1)
            END IF
            IF (( peig .GT. edge_energies(1,iZ) )) THEN
              j = 1
              sigma = (edge_a(1,iZ) + edge_b(1,iZ)/peig + edge_c(1,iZ)/a
     *        ux + edge_d(1,iZ)/aux1)/peig
            ELSE
              DO 5971 j=2,edge_number(iZ)
                IF((peig .GE. edge_energies(j,iZ)))GO TO5972
5971          CONTINUE
5972          CONTINUE
              sigma = edge_a(j,iZ) + gle*(edge_b(j,iZ) + gle*(edge_c(j,i
     *        Z) + gle*edge_d(j,iZ) ))
              sigma = Exp(sigma)
            END IF
            sigma = sigma * pz(medium,k)
            sigtot = sigtot + sigma
            probs(k) = sigma
            ints(k) = j
5961      CONTINUE
5962      CONTINUE
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          br = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          br = br*sigtot
          DO 5981 k=1,nne(medium)
            br = br - probs(k)
            IF((br .LE. 0))GO TO5982
5981      CONTINUE
5982      CONTINUE
          iZ = int( zelem(medium,k) + 0.5 )
          j = ints(k)
        END IF
        IF (( peig .LE. binding_energies(6,iZ) )) THEN
          iq(np) = -1
          e(np) = peig + prm
        ELSE
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          br = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          DO 5991 k=1,5
            IF (( peig .GT. binding_energies(k,iZ) )) THEN
              IF((br .LT. interaction_prob(k,iZ)))GO TO5992
              br = (br - interaction_prob(k,iZ))/(1-interaction_prob(k,i
     *        Z))
            END IF
5991      CONTINUE
5992      CONTINUE
          IF ((eadl_relax .AND. k .GT. 4)) THEN
            iq(np) = -1
            e(np) = peig + prm
          ELSE
            e_vac = binding_energies(k,iZ)
            e(np) = peig - e_vac + prm
            do_relax = .true.
            iq(np) = -1
          END IF
        END IF
      ELSE
        e(np) = peig + prm
        iq(np) = -1
      END IF
      IF (( iq(np) .EQ. -1 )) THEN
        IF ((IPHTER(IR(NP)).EQ.1)) THEN
          EELEC=E(NP)
          IF ((EELEC.GT.ECUT(IR(NP)))) THEN
            BETA=SQRT((EELEC-RM)*(EELEC+RM))/EELEC
            GAMMA=EELEC/RM
            ALPHA=0.5*GAMMA-0.5+1./GAMMA
            RATIO=BETA/ALPHA
6001        CONTINUE
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              RNPHT = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              RNPHT=2.*RNPHT-1.
              IF ((RATIO.LE.0.2)) THEN
                FKAPPA=RNPHT+0.5*RATIO*(1.-RNPHT)*(1.+RNPHT)
                IF (( gamma .LT. 100 )) THEN
                  COSTHE=(BETA+FKAPPA)/(1.+BETA*FKAPPA)
                ELSE
                  IF (( fkappa .GT. 0 )) THEN
                    costhe = 1 - (1-fkappa)*(gamma-3)/(2*(1+fkappa)*(gam
     *              ma-1)**3)
                  ELSE
                    COSTHE=(BETA+FKAPPA)/(1.+BETA*FKAPPA)
                  END IF
                END IF
                xi = (1+beta*fkappa)*gamma*gamma
              ELSE
                XI=GAMMA*GAMMA*(1.+ALPHA*(SQRT(1.+RATIO*(2.*RNPHT+RATIO)
     *          )-1.))
                COSTHE=(1.-1./XI)/BETA
              END IF
              SINTH2=MAX(0.,(1.-COSTHE)*(1.+COSTHE))
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              RNPHT2 = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              IF(RNPHT2.LE.0.5*(1.+GAMMA)*SINTH2*XI/GAMMA)GO TO6002
            GO TO 6001
6002        CONTINUE
            SINTHE=SQRT(SINTH2)
            CALL UPHI(2,1)
          END IF
        END IF
      END IF
      IF (( do_relax )) THEN
        call relax(e_vac,k,iZ)
      END IF
      IF (( EDEP .GT. 0 )) THEN
        IARG=4
        IF ((IAUSFL(IARG+1).NE.0)) THEN
          CALL AUSGAB(IARG)
        END IF
      END IF
      i_survived_RR = 0
      IF (( i_play_RR .EQ. 1 )) THEN
        IF (( prob_RR .LE. 0 )) THEN
          IF (( n_RR_warning .LT. 50 )) THEN
            n_RR_warning = n_RR_warning + 1
            WRITE(6,6010)prob_RR
6010        FORMAT('**** Warning, attempt to play Roussian Roulette with
     * prob_RR<=0! ',g14.6)
          END IF
        ELSE
          ip = NPold
6021      CONTINUE
            IF (( iq(ip) .NE. 0 )) THEN
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              rnno_RR = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              IF (( rnno_RR .LT. prob_RR )) THEN
                wt(ip) = wt(ip)/prob_RR
                ip = ip + 1
              ELSE
                i_survived_RR = i_survived_RR + 1
                IF ((ip .LT. np)) THEN
                  e(ip) = e(np)
                  iq(ip) = iq(np)
                  wt(ip) = wt(np)
                  u(ip) = u(np)
                  v(ip) = v(np)
                  w(ip) = w(np)
                END IF
                np = np-1
              END IF
            ELSE
              ip = ip+1
            END IF
            IF(((ip .GT. np)))GO TO6022
          GO TO 6021
6022      CONTINUE
          IF (( np .EQ. 0 )) THEN
            np = 1
            e(np) = 0
            iq(np) = 0
            wt(np) = 0
          END IF
        END IF
      END IF
      return
      end
      subroutine egs_shellwise_photo
      implicit none
      COMMON/BOUNDS/ECUT(1),PCUT(1),VACDST
      real*8 ECUT,  PCUT,  VACDST
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      COMMON/QDEBUG/QDEBUG
      LOGICAL QDEBUG
      COMMON/EDGE/binding_energies(30,100), interaction_prob(6,100), rel
     *axation_prob(39,100), edge_energies(16,100), edge_number(100), edg
     *e_a(16,100), edge_b(16,100), edge_c(16,100), edge_d(16,100), IEDGF
     *L(1),IPHTER(1)
      real*8 binding_energies,  interaction_prob,    relaxation_prob,  e
     *dge_energies,  edge_a,edge_b,edge_c,edge_d
      integer*2 IEDGFL,  IPHTER
      integer*4 edge_number
      COMMON/EPCONT/EDEP,EDEP_LOCAL,TSTEP,TUSTEP,USTEP,TVSTEP,VSTEP, RHO
     *F,EOLD,ENEW,EKE,ELKE,GLE,E_RANGE, x_final,y_final,z_final, u_final
     *,v_final,w_final, IDISC,IROLD,IRNEW,IAUSFL(35)
      DOUBLE PRECISION EDEP,  EDEP_LOCAL
      real*8 TSTEP,  TUSTEP,  USTEP,  VSTEP,  TVSTEP,  RHOF,  EOLD,  ENE
     *W,  EKE,  ELKE,  GLE,  E_RANGE, x_final,y_final,z_final,  u_final,
     *v_final,w_final
      integer*4 IDISC,  IROLD,  IRNEW,  IAUSFL
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/PHOTIN/ EBINDA(1), GE0(1),GE1(1), GMFP0(2000,1),GMFP1(2000,
     *1),GBR10(2000,1),GBR11(2000,1),GBR20(2000,1),GBR21(2000,1), RCO0(1
     *),RCO1(1), RSCT0(100,1),RSCT1(100,1), COHE0(2000,1),COHE1(2000,1),
     *  PHOTONUC0(2000,1),PHOTONUC1(2000,1), DPMFP, MPGEM(1,1), NGR(1)
      real*8 EBINDA,  GE0,GE1,  GMFP0,GMFP1,  GBR10,GBR11,  GBR20,GBR21,
     *  RCO0,RCO1,  RSCT0,RSCT1,  COHE0,COHE1,   PHOTONUC0,PHOTONUC1,  D
     *PMFP
      integer*4
     *                  MPGEM,
     *                          NGR
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/UPHIOT/THETA,SINTHE,COSTHE,SINPHI, COSPHI,PI,TWOPI,PI5D2
      real*8 THETA,  SINTHE,  COSTHE,  SINPHI,  COSPHI,  PI,TWOPI,PI5D2
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      common/x_options/eadl_relax,  mcdf_pe_xsections
      logical eadl_relax, mcdf_pe_xsections
      common/egs_vr/ e_max_rr(1),  prob_RR,  nbr_split,  i_play_RR,
     * i_survived_RR,
     *     n_RR_warning,                                        i_do_rr(
     *1)
      real*8          e_max_rr,prob_RR
      integer*4       nbr_split,i_play_RR,i_survived_RR,n_RR_warning
      integer*2     i_do_rr
      common/relax_data/ relax_first(3000),  relax_ntran(3000),  relax_s
     *tate(10000),  relax_prob(10000),  relax_atbin(10000),  relax_ntot
      real*8 relax_prob
      integer*4 relax_first, relax_ntran, relax_state, relax_atbin, rela
     *x_ntot
      common/pe_shell_data/ pe_xsection(500,100,0:16),  pe_elem_prob(500
     *,100,1),   pe_energy(500,100),  pe_zsorted(100,1), pe_be(100,16),
     * pe_nshell(100),  pe_zpos(100),  pe_nge(100),  pe_ne
      real*8 pe_be, pe_energy, pe_xsection, pe_elem_prob
      integer*4 pe_zsorted, pe_nshell, pe_zpos, pe_nge, pe_ne
      real*8 EELEC,  BETA,  GAMMA,  ALPHA,  RATIO,  RNPHT,  FKAPPA, XI,
     * SINTH2, RNPHT2
      DOUBLE PRECISION PEIG
      real*8 BR,  sigma,  aux,aux1,  probs(50),  sigtot,  e_vac,  rnno_R
     *R
      integer*4 IARG,  iZ,   irl,  ints(50),  j,ip,  n_warning,  k
      logical do_relax
      save n_warning
      real*8 slope, logE, int_prob
      integer*4 zpos, ibsearch
      data n_warning/0/
      NPold = NP
      PEIG=E(NP)
      irl = ir(np)
      do_relax = .false.
      IF (( peig .LT. 0.001 )) THEN
        IF (( n_warning .LT. 100 )) THEN
          n_warning = n_warning + 1
          write(i_log,*) ' Subroutine egs_shellwise_photo called with E
     *= ', peig,' which is below the current min. energy of ', 0.001,' k
     *eV! '
          write(i_log,*) ' Converting now this photon to an electron, '
          write(i_log,*) ' but you should check your code! '
        END IF
        iq(np) = -1
        e(np) = peig + prm
        return
      END IF
      edep = pzero
      IF (( iedgfl(irl) .NE. 0 )) THEN
        j = -1
        IF (( nne(medium) .EQ. 1 )) THEN
          iZ = int( zelem(medium,1) + 0.5 )
          zpos = pe_zpos(iZ)
          IF (( pe_nshell(zpos) .GT. 0)) THEN
            logE = log(peig)
            j = ibsearch(logE,pe_nge(zpos),pe_energy(1,zpos))
          END IF
        ELSE
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          br = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          logE = log(peig)
          DO 6031 k=nne(medium),1,-1
            iZ = int( zelem(medium,k) + 0.5 )
            zpos = pe_zpos(iZ)
            IF (( iZ .LT. 1 .OR. iZ .GT. 100 )) THEN
              write(i_log,*) ' Error in egs_shellwise_photo: '
              write(i_log,'(/a)') '***************** Error: '
              write(i_log,*) '   Atomic number of element ',k, ' in medi
     *um ',medium,' is not between 1 and ',100
              write(i_log,'(/a)') '***************** Quiting now.'
              call exit(1)
            END IF
            j = ibsearch(logE,pe_nge(zpos),pe_energy(1,zpos))
            slope = pe_elem_prob(j+1,k,medium) - pe_elem_prob(j,k,medium
     *      )
            slope = slope/(pe_energy(j+1,zpos)-pe_energy(j,zpos))
            int_prob = pe_elem_prob(j,k,medium)+slope*(logE-pe_energy(j,
     *      zpos))
            br = br - exp(int_prob)
            IF((br .LE. 0))GO TO6032
6031      CONTINUE
6032      CONTINUE
        END IF
        IF (( peig .LT. pe_be(zpos,pe_nshell(zpos)) .OR. pe_nshell(zpos)
     *   .EQ. 0 )) THEN
          iq(np) = -1
          e(np) = peig + prm
        ELSE
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          br = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          sigtot = 0
          DO 6041 k=1,pe_nshell(zpos)
            IF (( peig .GT. pe_be(zpos,k) )) THEN
              slope = pe_xsection(j+1,zpos,k) - pe_xsection(j,zpos,k)
              slope = slope/(pe_energy(j+1,zpos)-pe_energy(j,zpos))
              int_prob=pe_xsection(j,zpos,k)+slope*(logE-pe_energy(j,zpo
     *        s))
              br = br - exp(int_prob)
              sigtot = sigtot + exp(int_prob)
              IF((br .LE. 0))GO TO6042
            END IF
6041      CONTINUE
6042      CONTINUE
          IF ((k .GT. pe_nshell(zpos))) THEN
            iq(np) = -1
            e(np) = peig + prm
          ELSE
            e_vac = pe_be(zpos,k)
            e(np) = peig - e_vac + prm
            do_relax = .true.
            iq(np) = -1
          END IF
        END IF
      ELSE
        e(np) = peig + prm
        iq(np) = -1
      END IF
      IF (( iq(np) .EQ. -1 )) THEN
        IF ((IPHTER(IR(NP)).EQ.1)) THEN
          EELEC=E(NP)
          IF ((EELEC.GT.ECUT(IR(NP)))) THEN
            BETA=SQRT((EELEC-RM)*(EELEC+RM))/EELEC
            GAMMA=EELEC/RM
            ALPHA=0.5*GAMMA-0.5+1./GAMMA
            RATIO=BETA/ALPHA
6051        CONTINUE
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              RNPHT = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              RNPHT=2.*RNPHT-1.
              IF ((RATIO.LE.0.2)) THEN
                FKAPPA=RNPHT+0.5*RATIO*(1.-RNPHT)*(1.+RNPHT)
                IF (( gamma .LT. 100 )) THEN
                  COSTHE=(BETA+FKAPPA)/(1.+BETA*FKAPPA)
                ELSE
                  IF (( fkappa .GT. 0 )) THEN
                    costhe = 1 - (1-fkappa)*(gamma-3)/(2*(1+fkappa)*(gam
     *              ma-1)**3)
                  ELSE
                    COSTHE=(BETA+FKAPPA)/(1.+BETA*FKAPPA)
                  END IF
                END IF
                xi = (1+beta*fkappa)*gamma*gamma
              ELSE
                XI=GAMMA*GAMMA*(1.+ALPHA*(SQRT(1.+RATIO*(2.*RNPHT+RATIO)
     *          )-1.))
                COSTHE=(1.-1./XI)/BETA
              END IF
              SINTH2=MAX(0.,(1.-COSTHE)*(1.+COSTHE))
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              RNPHT2 = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              IF(RNPHT2.LE.0.5*(1.+GAMMA)*SINTH2*XI/GAMMA)GO TO6052
            GO TO 6051
6052        CONTINUE
            SINTHE=SQRT(SINTH2)
            CALL UPHI(2,1)
          END IF
        END IF
      END IF
      IF (( do_relax )) THEN
        call egs_eadl_relax(iZ,k)
      END IF
      IF (( EDEP .GT. 0 )) THEN
        IARG=4
        IF ((IAUSFL(IARG+1).NE.0)) THEN
          CALL AUSGAB(IARG)
        END IF
      END IF
      i_survived_RR = 0
      IF (( i_play_RR .EQ. 1 )) THEN
        IF (( prob_RR .LE. 0 )) THEN
          IF (( n_RR_warning .LT. 50 )) THEN
            n_RR_warning = n_RR_warning + 1
            WRITE(6,6060)prob_RR
6060        FORMAT('**** Warning, attempt to play Roussian Roulette with
     * prob_RR<=0! ',g14.6)
          END IF
        ELSE
          ip = NPold
6071      CONTINUE
            IF (( iq(ip) .NE. 0 )) THEN
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              rnno_RR = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              IF (( rnno_RR .LT. prob_RR )) THEN
                wt(ip) = wt(ip)/prob_RR
                ip = ip + 1
              ELSE
                i_survived_RR = i_survived_RR + 1
                IF ((ip .LT. np)) THEN
                  e(ip) = e(np)
                  iq(ip) = iq(np)
                  wt(ip) = wt(np)
                  u(ip) = u(np)
                  v(ip) = v(np)
                  w(ip) = w(np)
                END IF
                np = np-1
              END IF
            ELSE
              ip = ip+1
            END IF
            IF(((ip .GT. np)))GO TO6072
          GO TO 6071
6072      CONTINUE
          IF (( np .EQ. 0 )) THEN
            np = 1
            e(np) = 0
            iq(np) = 0
            wt(np) = 0
          END IF
        END IF
      END IF
      return
      end
      subroutine egs_read_shellwise_pe
      implicit none
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      COMMON/EDGE/binding_energies(30,100), interaction_prob(6,100), rel
     *axation_prob(39,100), edge_energies(16,100), edge_number(100), edg
     *e_a(16,100), edge_b(16,100), edge_c(16,100), edge_d(16,100), IEDGF
     *L(1),IPHTER(1)
      real*8 binding_energies,  interaction_prob,    relaxation_prob,  e
     *dge_energies,  edge_a,edge_b,edge_c,edge_d
      integer*2 IEDGFL,  IPHTER
      integer*4 edge_number
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/PHOTIN/ EBINDA(1), GE0(1),GE1(1), GMFP0(2000,1),GMFP1(2000,
     *1),GBR10(2000,1),GBR11(2000,1),GBR20(2000,1),GBR21(2000,1), RCO0(1
     *),RCO1(1), RSCT0(100,1),RSCT1(100,1), COHE0(2000,1),COHE1(2000,1),
     *  PHOTONUC0(2000,1),PHOTONUC1(2000,1), DPMFP, MPGEM(1,1), NGR(1)
      real*8 EBINDA,  GE0,GE1,  GMFP0,GMFP1,  GBR10,GBR11,  GBR20,GBR21,
     *  RCO0,RCO1,  RSCT0,RSCT1,  COHE0,COHE1,   PHOTONUC0,PHOTONUC1,  D
     *PMFP
      integer*4
     *                  MPGEM,
     *                          NGR
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      common/x_options/eadl_relax,  mcdf_pe_xsections
      logical eadl_relax, mcdf_pe_xsections
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common/pe_shell_data/ pe_xsection(500,100,0:16),  pe_elem_prob(500
     *,100,1),   pe_energy(500,100),  pe_zsorted(100,1), pe_be(100,16),
     * pe_nshell(100),  pe_zpos(100),  pe_nge(100),  pe_ne
      real*8 pe_be, pe_energy, pe_xsection, pe_elem_prob
      integer*4 pe_zsorted, pe_nshell, pe_zpos, pe_nge, pe_ne
      integer*4 lnblnk1,egs_get_unit,pe_sw_unit,ounit,egs_open_file
      integer*4 sorted(100),i,j,k,l,m
      real*8 z_sorted(100),pz_sorted(100)
      real*8 rest_xs(500,100)
      real*8 tmp_e(500,16), tmp_xs(500,16)
      real*8 new_e(500),deltaEb,slope
      integer*4 zread(100),ib(16),ibsearch
      character data_dir*128,pe_sw_file*144
      integer*4 medio,iZ,iZpos,egs_read_int,pos,curr_rec
      real*4 egs_read_real,e_r, e_old,sigma_r
      integer*2 nz, egs_read_short,ish, i_nshell,i_nge
      logical is_open, is_there, shift_required
      character*3 labels(16)
      data labels/'  K',' L1',' L2',' L3', ' M1',' M2',' M3',' M4',' M5'
     *, ' N1',' N2',' N3',' N4',' N5',' N6',' N7'/
      write(i_log,'(/a$)') ' Reading renormalized photoelectric cross se
     *ctions ......'
      data_dir = hen_house(:lnblnk1(hen_house)) // 'data' // '/'
      pe_sw_file = data_dir(:lnblnk1(data_dir)) // 'photo_shellwise.data
     *'
      pe_sw_unit = egs_get_unit(0)
      IF (( pe_sw_unit .LT. 1 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'egs_init_shellwise_pe: failed to get a free Fort
     *ran I/O unit'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      open(pe_sw_unit,file=pe_sw_file,status='old', form='UNFORMATTED',A
     *CCESS='direct',recl=1, err=6080)
      GOTO 6090
6080  write(i_log,'(/a)') '***************** Error: '
      write(i_log,'(2a)') 'egs_init_shellwise_pe: failed to open ', pe_s
     *w_file
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
6090  is_open = .true.
      DO 6101 medio=1,nmed
        DO 6111 i=1,nne(medio)
          pe_nshell(i*medio) = 0
          pe_nge(i*medio) = 0
          pe_zsorted(i,medio) = 0
6111    CONTINUE
6112    CONTINUE
6101  CONTINUE
6102  CONTINUE
      DO 6121 l=1,100
        pe_zpos(l) = -1
        DO 6131 k=1,500
          pe_energy(k,l) = 0.0
          DO 6141 m=1,16
            pe_xsection(k,l,m) = 0.0
6141      CONTINUE
6142      CONTINUE
6131    CONTINUE
6132    CONTINUE
        DO 6151 k=1,16
          pe_be(l,k) = -99
6151    CONTINUE
6152    CONTINUE
6121  CONTINUE
6122  CONTINUE
      curr_rec = 1
      iZpos = 0
      nz = egs_read_short(pe_sw_unit,curr_rec)
      DO 6161 medio=1,nmed
        DO 6171 i=1,nne(medio)
          z_sorted(i) = zelem(medio,i)
6171    CONTINUE
6172    CONTINUE
        call egs_heap_sort(nne(medio),z_sorted,sorted)
        DO 6181 i=1,nne(medio)
          pe_zsorted(i,medio) = z_sorted(i)
6181    CONTINUE
6182    CONTINUE
        DO 6191 i=1,nne(medio)
          iZ = z_sorted(i)
          is_there = .false.
          DO 6201 j=1,medio-1
            DO 6211 k=1,nne(j)
              IF (( iZ .EQ. pe_zsorted(k,j) )) THEN
                is_there = .true.
                GO TO6212
              END IF
6211        CONTINUE
6212        CONTINUE
6201      CONTINUE
6202      CONTINUE
          IF((is_there))GO TO6191
          iZpos = iZpos + 1
          zread(iZpos) = iZ
          pe_zpos(iZ) = iZpos
          pos = 3 + (iZ-1)*4
          curr_rec = egs_read_int(pe_sw_unit,pos) + 1
          i_nge = egs_read_short(pe_sw_unit,curr_rec)
          i_nshell = egs_read_short(pe_sw_unit,curr_rec)
          pe_nge(iZpos) = i_nge
          pe_nshell(iZpos) = i_nshell
          e_old = -1.0
          ish = 0
          DO 6221 j=1,i_nge
            e_r = egs_read_real(pe_sw_unit,curr_rec)
            sigma_r = egs_read_real(pe_sw_unit,curr_rec)
            pe_energy(j,iZpos) = e_r
            pe_xsection(j,iZpos,0) = sigma_r
            rest_xs(j,iZpos) = sigma_r
            DO 6231 k=1,i_nshell
              sigma_r = egs_read_real(pe_sw_unit,curr_rec)
              pe_xsection(j,iZpos,k) = sigma_r
              rest_xs(j,iZpos) = rest_xs(j,iZpos) - sigma_r
6231        CONTINUE
6232        CONTINUE
            IF ((e_r - e_old .LT. 1e-15)) THEN
              pe_be(iZpos,i_nshell-ish) = e_r
              ish = ish + 1
            END IF
            e_old = e_r
6221      CONTINUE
6222      CONTINUE
6191    CONTINUE
6192    CONTINUE
6161  CONTINUE
6162  CONTINUE
      pe_ne = iZpos
      DO 6241 i=1,pe_ne
        iZ = zread(i)
        IF ((pe_nshell(i) .EQ. 0)) THEN
          DO 6251 j=1,pe_nge(i)
            pe_energy(j,i) = log(pe_energy(j,i))
6251      CONTINUE
6252      CONTINUE
          GO TO6241
        END IF
        DO 6261 l=1,pe_nshell(i)
          IF (( pe_be(i,l) .NE. binding_energies(l,iZ))) THEN
            shift_required = .true.
            deltaEb = binding_energies(l,iZ)-pe_be(i,l)
          ELSE
            shift_required =.false.
          END IF
          is_there = .false.
          DO 6271 j=1,pe_nge(i)
            tmp_e(j,l) = pe_energy(j,i)
            tmp_xs(j,l) = pe_xsection(j,i,l)
            IF (( shift_required .AND. pe_energy(j,i) .GE. pe_be(i,l) ))
     *       THEN
              tmp_e(j,l) = tmp_e(j,l) + deltaEb
              IF ((pe_energy(j,i) .EQ. pe_be(i,l) .AND. .NOT.is_there))
     *        THEN
                ib(l) = j
                is_there = .true.
              END IF
              IF ((l .EQ. 1)) THEN
                new_e(j) = tmp_e(j,l)
              ELSE IF((j .LT. ib(l-1))) THEN
                new_e(j) = tmp_e(j,l)
              END IF
            END IF
6271      CONTINUE
6272      CONTINUE
          pe_be(i,l) = binding_energies(l,iZ)
6261    CONTINUE
6262    CONTINUE
        DO 6281 l=2,pe_nshell(i)
          DO 6291 j=1,pe_nge(i)
            IF (( new_e(j) .GE. pe_be(i,l-1) )) THEN
              m = ibsearch(new_e(j),pe_nge(i),tmp_e(1,l))
              slope = log(tmp_xs(m+1,l)/tmp_xs(m,l))
              slope = slope/log(tmp_e(m+1,l)/tmp_e(m,l))
              pe_xsection(j,i,l) = log(tmp_xs(m,l))
              pe_xsection(j,i,l) = pe_xsection(j,i,l) + slope*log(new_e(
     *        j)/tmp_e(m,l))
              pe_xsection(j,i,l) = exp(pe_xsection(j,i,l))
            END IF
6291      CONTINUE
6292      CONTINUE
6281    CONTINUE
6282    CONTINUE
        DO 6301 j=1,pe_nge(i)
          IF (( j .LT. ib(pe_nshell(i)))) THEN
            new_e(j) = pe_energy(j,i)
          END IF
          m = ibsearch(new_e(j),pe_nge(i),pe_energy(1,i))
          slope = log(rest_xs(m+1,i)/rest_xs(m,i))
          slope = slope/log(pe_energy(m+1,i)/pe_energy(m,i))
          pe_xsection(j,i,0) = log(rest_xs(m,i))
          pe_xsection(j,i,0) = pe_xsection(j,i,0) + slope*log(new_e(j)/p
     *    e_energy(m,i))
          pe_xsection(j,i,0) = exp(pe_xsection(j,i,0))
          DO 6311 l=1,pe_nshell(i)
            pe_xsection(j,i,0) = pe_xsection(j,i,0) + pe_xsection(j,i,l)
6311      CONTINUE
6312      CONTINUE
6301    CONTINUE
6302    CONTINUE
        DO 6321 j=1,pe_nge(i)
          pe_energy(j,i) = log(new_e(j))
          DO 6331 l=1,pe_nshell(i)
            pe_xsection(j,i,l) = log(pe_xsection(j,i,l)/pe_xsection(j,i,
     *      0))
6331      CONTINUE
6332      CONTINUE
6321    CONTINUE
6322    CONTINUE
6241  CONTINUE
6242  CONTINUE
      write(i_log,'(a/)') ' done'
      IF((is_open))close(pe_sw_unit)
      return
      end
      SUBROUTINE RELAX(energy,n,iZ)
      implicit none
      integer*4 n,iZ
      real*8 energy
      integer*4 vac_array(50),  n_vac,  shell
      integer*4 final,finala,  final1,final2,   iql,  irl
      integer*4 first_transition(5), last_transition(5)
      integer*4 final_state(39)
      integer*4 k, np_old, ip, iarg
      real*8 e_array(50),  Ei,Ef,  Ex,  eta,  e_check,  min_E,ekcut,pkcu
     *t,elcut
      real*8 xphi,yphi,xphi2,yphi2,rhophi2, cphi,sphi
      COMMON/EPCONT/EDEP,EDEP_LOCAL,TSTEP,TUSTEP,USTEP,TVSTEP,VSTEP, RHO
     *F,EOLD,ENEW,EKE,ELKE,GLE,E_RANGE, x_final,y_final,z_final, u_final
     *,v_final,w_final, IDISC,IROLD,IRNEW,IAUSFL(35)
      DOUBLE PRECISION EDEP,  EDEP_LOCAL
      real*8 TSTEP,  TUSTEP,  USTEP,  VSTEP,  TVSTEP,  RHOF,  EOLD,  ENE
     *W,  EKE,  ELKE,  GLE,  E_RANGE, x_final,y_final,z_final,  u_final,
     *v_final,w_final
      integer*4 IDISC,  IROLD,  IRNEW,  IAUSFL
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/BOUNDS/ECUT(1),PCUT(1),VACDST
      real*8 ECUT,  PCUT,  VACDST
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      COMMON/EDGE/binding_energies(30,100), interaction_prob(6,100), rel
     *axation_prob(39,100), edge_energies(16,100), edge_number(100), edg
     *e_a(16,100), edge_b(16,100), edge_c(16,100), edge_d(16,100), IEDGF
     *L(1),IPHTER(1)
      real*8 binding_energies,  interaction_prob,    relaxation_prob,  e
     *dge_energies,  edge_a,edge_b,edge_c,edge_d
      integer*2 IEDGFL,  IPHTER
      integer*4 edge_number
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      common/relax_data/ relax_first(3000),  relax_ntran(3000),  relax_s
     *tate(10000),  relax_prob(10000),  relax_atbin(10000),  relax_ntot
      real*8 relax_prob
      integer*4 relax_first, relax_ntran, relax_state, relax_atbin, rela
     *x_ntot
      common/x_options/eadl_relax,  mcdf_pe_xsections
      logical eadl_relax, mcdf_pe_xsections
      common/user_relax/ u_relax,ish_relax,iZ_relax
      real*8 u_relax
      integer*4 ish_relax, iZ_relax
      data first_transition/1,20,27,33,38/
      data last_transition/19,26,32,37,39/
      data final_state/  4,3,5,6,  202,302,402,404,403,303,  502,503,504
     *,602,603,604,  505,605,606,  13,14,  5,6,  505,605,606,  14,  5,6,
     *  505,605,606,  5,6,  505,605,606,  6,  606/
      save first_transition,last_transition,final_state
      IF ((eadl_relax)) THEN
        call egs_eadl_relax(iZ,n)
        return
      END IF
      IF (( n .LT. 1 .OR. n .GT. 6 )) THEN
        return
      END IF
      iz_relax = iZ
      irl = ir(np)
      ekcut = ecut(irl)-rm
      pkcut = pcut(irl)
      min_E = 0.001
      IF (( energy .LE. min_E )) THEN
        edep = edep + energy
        edep_local = energy
        IARG=34
        IF ((IAUSFL(IARG+1).NE.0)) THEN
          CALL AUSGAB(IARG)
        END IF
        return
      END IF
      n_vac = 1
      vac_array(n_vac) = n
      np_old = np
      e_check = 0
      e_array(n_vac) = energy
6340  CONTINUE
6341    CONTINUE
        shell = vac_array(n_vac)
        Ei = e_array(n_vac)
        n_vac = n_vac - 1
        IF (( Ei .LE. min_E )) THEN
          edep = edep + Ei
          edep_local = Ei
          IARG=34
          IF ((IAUSFL(IARG+1).NE.0)) THEN
            CALL AUSGAB(IARG)
          END IF
          IF((n_vac .GT. 0))goto 6340
          GO TO6342
        END IF
        ish_relax = shell
        u_relax = Ei
        IF (( shell .EQ. 6 )) THEN
          IF (( Ei .GT. ekcut )) THEN
            np = np + 1
            IF (( np .GT. 2 )) THEN
              write(i_log,'(/a)') '***************** Error: '
              write(i_log,'(//,3a,/,2(a,i9))') ' In subroutine ','RELAX'
     *        , ' stack size exceeded! ',' $MAXSTACK = ',2,' np = ',np
              write(i_log,'(/a)') '***************** Quiting now.'
              call exit(1)
            END IF
            e(np) = Ei + prm
            iq(np) = -1
            X(np)=X(np-1)
            Y(np)=Y(np-1)
            Z(np)=Z(np-1)
            IR(np)=IR(np-1)
            WT(np)=WT(np-1)
            DNEAR(np)=DNEAR(np-1)
            LATCH(np)=LATCH(np-1)
            IF (( rng_seed .GT. 24 )) THEN
              call ranlux(rng_array)
              rng_seed = 1
            END IF
            eta = rng_array(rng_seed)
            rng_seed = rng_seed + 1
            eta = 2*eta - 1
            w(np) = eta
            eta = (1-eta)*(1+eta)
            IF (( eta .GT. 1e-20 )) THEN
              eta = Sqrt(eta)
6351          CONTINUE
                IF (( rng_seed .GT. 24 )) THEN
                  call ranlux(rng_array)
                  rng_seed = 1
                END IF
                xphi = rng_array(rng_seed)
                rng_seed = rng_seed + 1
                xphi = 2*xphi - 1
                xphi2 = xphi*xphi
                IF (( rng_seed .GT. 24 )) THEN
                  call ranlux(rng_array)
                  rng_seed = 1
                END IF
                yphi = rng_array(rng_seed)
                rng_seed = rng_seed + 1
                yphi2 = yphi*yphi
                rhophi2 = xphi2 + yphi2
                IF(rhophi2.LE.1)GO TO6352
              GO TO 6351
6352          CONTINUE
              rhophi2 = 1/rhophi2
              cphi = (xphi2 - yphi2)*rhophi2
              sphi = 2*xphi*yphi*rhophi2
              u(np) = eta*cphi
              v(np) = eta*sphi
            ELSE
              u(np) = 0
              v(np) = 0
              w(np) = 1
            END IF
            IARG=27
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
          ELSE
            edep = edep + Ei
            edep_local = Ei
            IARG=34
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
          END IF
          IF((n_vac .GT. 0))goto 6340
          GO TO6342
        END IF
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        eta = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        DO 6361 k=first_transition(shell),last_transition(shell)-1
          eta = eta - relaxation_prob(k,iZ)
          IF((eta .LE. 0))GO TO6362
6361    CONTINUE
6362    CONTINUE
        final = final_state(k)
        finala = final
        IF (( final .LT. 100 )) THEN
          IF (( final .LT. 10 )) THEN
            iql = 0
            elcut = pkcut
          ELSE
            final = final - 10
            iql = -1
            elcut = ekcut
          END IF
          Ef = binding_energies(final,iZ)
          Ex = Ei - Ef
          n_vac = n_vac + 1
          vac_array(n_vac) = final
          e_array(n_vac) = Ef
        ELSE
          final1 = final/100
          final2 = final - final1*100
          n_vac = n_vac + 1
          vac_array(n_vac) = final1
          e_array(n_vac) = binding_energies(final1,iZ)
          n_vac = n_vac + 1
          vac_array(n_vac) = final2
          e_array(n_vac) = binding_energies(final2,iZ)
          iql = -1
          Ex = Ei - e_array(n_vac) - e_array(n_vac-1)
          elcut = ekcut
        END IF
        IF (( Ex .LE. elcut )) THEN
          edep = edep + Ex
          IF (( finala .LT. 10 )) THEN
            edep_local = Ex
            IARG=33
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
          ELSE
            edep_local = Ex
            IARG=34
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
          END IF
        ELSE
          np = np + 1
          IF (( np .GT. 2 )) THEN
            write(i_log,'(/a)') '***************** Error: '
            write(i_log,'(//,3a,/,2(a,i9))') ' In subroutine ','RELAX',
     *      ' stack size exceeded! ',' $MAXSTACK = ',2,' np = ',np
            write(i_log,'(/a)') '***************** Quiting now.'
            call exit(1)
          END IF
          iq(np) = iql
          IF (( iql .EQ. 0 )) THEN
            e(np) = Ex
          ELSE
            e(np) = Ex + rm
          END IF
          X(np)=X(np-1)
          Y(np)=Y(np-1)
          Z(np)=Z(np-1)
          IR(np)=IR(np-1)
          WT(np)=WT(np-1)
          DNEAR(np)=DNEAR(np-1)
          LATCH(np)=LATCH(np-1)
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          eta = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          eta = 2*eta - 1
          w(np) = eta
          eta = (1-eta)*(1+eta)
          IF (( eta .GT. 1e-20 )) THEN
            eta = Sqrt(eta)
6371        CONTINUE
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              xphi = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              xphi = 2*xphi - 1
              xphi2 = xphi*xphi
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              yphi = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              yphi2 = yphi*yphi
              rhophi2 = xphi2 + yphi2
              IF(rhophi2.LE.1)GO TO6372
            GO TO 6371
6372        CONTINUE
            rhophi2 = 1/rhophi2
            cphi = (xphi2 - yphi2)*rhophi2
            sphi = 2*xphi*yphi*rhophi2
            u(np) = eta*cphi
            v(np) = eta*sphi
          ELSE
            u(np) = 0
            v(np) = 0
            w(np) = 1
          END IF
          IF (( finala .LT. 10 )) THEN
            IARG=25
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
          ELSE IF(( finala .LT. 100 )) THEN
            IARG=26
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
          ELSE
            IARG=27
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
          END IF
        END IF
      GO TO 6341
6342  CONTINUE
      return
      end
      subroutine egs_init_relax
      implicit none
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      COMMON/EDGE/binding_energies(30,100), interaction_prob(6,100), rel
     *axation_prob(39,100), edge_energies(16,100), edge_number(100), edg
     *e_a(16,100), edge_b(16,100), edge_c(16,100), edge_d(16,100), IEDGF
     *L(1),IPHTER(1)
      real*8 binding_energies,  interaction_prob,    relaxation_prob,  e
     *dge_energies,  edge_a,edge_b,edge_c,edge_d
      integer*2 IEDGFL,  IPHTER
      integer*4 edge_number
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/PHOTIN/ EBINDA(1), GE0(1),GE1(1), GMFP0(2000,1),GMFP1(2000,
     *1),GBR10(2000,1),GBR11(2000,1),GBR20(2000,1),GBR21(2000,1), RCO0(1
     *),RCO1(1), RSCT0(100,1),RSCT1(100,1), COHE0(2000,1),COHE1(2000,1),
     *  PHOTONUC0(2000,1),PHOTONUC1(2000,1), DPMFP, MPGEM(1,1), NGR(1)
      real*8 EBINDA,  GE0,GE1,  GMFP0,GMFP1,  GBR10,GBR11,  GBR20,GBR21,
     *  RCO0,RCO1,  RSCT0,RSCT1,  COHE0,COHE1,   PHOTONUC0,PHOTONUC1,  D
     *PMFP
      integer*4
     *                  MPGEM,
     *                          NGR
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      common/x_options/eadl_relax,  mcdf_pe_xsections
      logical eadl_relax, mcdf_pe_xsections
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common/relax_data/ relax_first(3000),  relax_ntran(3000),  relax_s
     *tate(10000),  relax_prob(10000),  relax_atbin(10000),  relax_ntot
      real*8 relax_prob
      integer*4 relax_first, relax_ntran, relax_state, relax_atbin, rela
     *x_ntot
      common/shell_data/ shell_be(3000),  shell_type(3000),  shell_num(3
     *000),  shell_Z(3000),  shell_eadl(100,30),  shell_ntot
      real*8 shell_be
      integer*4 shell_type,shell_Z,shell_ntot,shell_num,shell_eadl
      integer*4 lnblnk1,egs_get_unit,relax_unit,ounit,egs_open_file
      integer*4 sorted(100),i,j,k,k1,k2,m
      real*8 z_sorted(100),pz_sorted(100)
      character data_dir*128,relax_file*144
      integer*4 ish,medio,iZ,ntran
      real*8 Ec, Pc, tmp, min_be, sumw,Ex
      logical is_open, is_there
      real*8 wtmp(300)
      integer*4 itmp(300)
      integer*4 pos, curr_rec, sh_eadl
      integer*4 nz, nshell, tr_type
      integer*4 ttype
      real*4 be_r, prob_r
      DO 6381 iZ=1,100
        DO 6391 k=1,30
          shell_eadl(iZ,k) = -1
6391    CONTINUE
6392    CONTINUE
6381  CONTINUE
6382  CONTINUE
      min_be = 0.001
      write(i_log,'(/a)') ' Reading EADL relaxation data ......'
      data_dir = hen_house(:lnblnk1(hen_house)) // 'data' // '/'
      relax_file = data_dir(:lnblnk1(data_dir)) // 'relax.data'
      relax_unit = egs_get_unit(0)
      IF (( relax_unit .LT. 1 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'egs_init_relax: failed to get a free Fortran I/O
     * unit'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      open(relax_unit,file=relax_file,status='old', form='UNFORMATTED',A
     *CCESS='direct',recl=4, err=6400)
      GOTO 6410
6400  write(i_log,'(/a)') '***************** Error: '
      write(i_log,'(2a)') 'egs_init_relax: failed to open ', relax_file
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
6410  is_open = .true.
      curr_rec = 1
      read(relax_unit,rec=curr_rec) nz
      shell_ntot = 0
      relax_ntot = 0
      DO 6421 medio=1,nmed
        DO 6431 i=1,nne(medio)
          z_sorted(i) = zelem(medio,i)
6431    CONTINUE
6432    CONTINUE
        call egs_heap_sort(nne(medio),z_sorted,sorted)
        DO 6441 i=1,nne(medio)
          iZ = z_sorted(i)
          is_there = .false.
          DO 6451 j=1,shell_ntot
            IF (( iZ .EQ. shell_Z(j) )) THEN
              is_there = .true.
              GO TO6452
            END IF
6451      CONTINUE
6452      CONTINUE
          IF((is_there))GO TO6441
          pos = iZ + 1
          read(relax_unit,rec=pos) curr_rec
          read(relax_unit,rec=curr_rec) nshell
          IF (( shell_ntot + nshell .GT. 3000 )) THEN
            write(i_log,'(/a)') '***************** Error: '
            write(i_log,'(a,i5,a/,a//)') ' Too many shells to fit in the
     * list: ', shell_ntot + nshell,' (at least).', ' Increase the param
     *eter $MAXSHELL and retry '
            write(i_log,'(/a)') '***************** Quiting now.'
            call exit(1)
          END IF
          write(i_log,'(a,i3,a,i2,a)') '  Z = ',iZ,' has ',nshell,' shel
     *ls'
          DO 6461 ish=shell_ntot+1,shell_ntot+nshell
            curr_rec = curr_rec+1
            read(relax_unit,rec=curr_rec) shell_type(ish)
            curr_rec = curr_rec+1
            read(relax_unit,rec=curr_rec) ntran
            curr_rec = curr_rec+1
            read(relax_unit,rec=curr_rec) be_r
            shell_be(ish) = be_r
            shell_Z(ish) = iZ
            shell_num(ish) = ish - shell_ntot
            shell_eadl(iZ,shell_num(ish)) = ish
            IF ((binding_energies(shell_num(ish),iZ) .GT. 0)) THEN
              shell_be(ish) = binding_energies(shell_num(ish),iZ)
            ELSE IF(( photon_xsections .EQ. 'epdl' )) THEN
              binding_energies(shell_num(ish),iZ) = shell_be(ish)
            END IF
            DO 6471 k=1,ntran
              curr_rec = curr_rec+1
              read(relax_unit,rec=curr_rec) itmp(k)
              curr_rec = curr_rec+1
              read(relax_unit,rec=curr_rec) prob_r
              wtmp(k)=prob_r
              IF ((itmp(k).LT.64)) THEN
                itmp(k) = itmp(k) + 1
              ELSE
                itmp(k) = itmp(k) + 65
              END IF
6471        CONTINUE
6472        CONTINUE
            IF (( shell_be(ish) .LT. min_be )) THEN
              relax_first(ish) = -1
              relax_ntran(ish) = -1
            ELSE
              sumw = 0
              DO 6481 k=1,ntran
                sumw = sumw + wtmp(k)
6481          CONTINUE
6482          CONTINUE
              IF (( sumw .GT. 1 )) THEN
                DO 6491 k=1,ntran
                  wtmp(k) = wtmp(k)/sumw
6491            CONTINUE
6492            CONTINUE
              ELSE IF(( sumw .LT. 1 )) THEN
                ntran = ntran + 1
                itmp(ntran) = -1
                wtmp(ntran) = 1-sumw
              END IF
              IF (( relax_ntot + ntran .GT. 10000 )) THEN
                write(i_log,'(/a)') '***************** Error: '
                write(i_log,'(a,i5,a/,a/)') ' Too many relaxation transi
     *tions: ', relax_ntot + ntran,' (at least).', ' Increase $MAXRELAX
     *and retry '
                write(i_log,'(/a)') '***************** Quiting now.'
                call exit(1)
              END IF
              relax_first(ish) = relax_ntot+1
              relax_ntran(ish) = ntran
              call prepare_alias_histogram(ntran,wtmp, relax_atbin(relax
     *        _ntot+1))
              DO 6501 k=1,ntran
                j = relax_ntot + k
                relax_state(j) = itmp(k)
                relax_prob(j) = wtmp(k)
6501          CONTINUE
6502          CONTINUE
              relax_ntot = relax_ntot + ntran
            END IF
6461      CONTINUE
6462      CONTINUE
          shell_ntot = shell_ntot + nshell
6441    CONTINUE
6442    CONTINUE
6421  CONTINUE
6422  CONTINUE
      write(i_log,'(a/)') ' ...... Done.'
      IF((is_open))close(relax_unit)
      return
      stop
      end
      subroutine egs_eadl_relax(iZ, shell_egs)
      implicit none
      common/relax_data/ relax_first(3000),  relax_ntran(3000),  relax_s
     *tate(10000),  relax_prob(10000),  relax_atbin(10000),  relax_ntot
      real*8 relax_prob
      integer*4 relax_first, relax_ntran, relax_state, relax_atbin, rela
     *x_ntot
      common/relax_for_user/ rfu_E0,  rfu_E,  rfu_Z,  rfu_j0,  rfu_n0,
     *rfu_t0,  rfu_j,  rfu_n,  rfu_t
      integer*4 rfu_Z,rfu_j0,rfu_n0,rfu_t0,rfu_j,rfu_n,rfu_t
      real*8 rfu_E0,rfu_E
      common/shell_data/ shell_be(3000),  shell_type(3000),  shell_num(3
     *000),  shell_Z(3000),  shell_eadl(100,30),  shell_ntot
      real*8 shell_be
      integer*4 shell_type,shell_Z,shell_ntot,shell_num,shell_eadl
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/EPCONT/EDEP,EDEP_LOCAL,TSTEP,TUSTEP,USTEP,TVSTEP,VSTEP, RHO
     *F,EOLD,ENEW,EKE,ELKE,GLE,E_RANGE, x_final,y_final,z_final, u_final
     *,v_final,w_final, IDISC,IROLD,IRNEW,IAUSFL(35)
      DOUBLE PRECISION EDEP,  EDEP_LOCAL
      real*8 TSTEP,  TUSTEP,  USTEP,  VSTEP,  TVSTEP,  RHOF,  EOLD,  ENE
     *W,  EKE,  ELKE,  GLE,  E_RANGE, x_final,y_final,z_final,  u_final,
     *v_final,w_final
      integer*4 IDISC,  IROLD,  IRNEW,  IAUSFL
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      COMMON/UPHIOT/THETA,SINTHE,COSTHE,SINPHI, COSPHI,PI,TWOPI,PI5D2
      real*8 THETA,  SINTHE,  COSTHE,  SINPHI,  COSPHI,  PI,TWOPI,PI5D2
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      COMMON/BOUNDS/ECUT(1),PCUT(1),VACDST
      real*8 ECUT,  PCUT,  VACDST
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      COMMON/MISC/  DUNIT,KMPI,KMPO,RHOR(1),MED(1),IRAYLR(1),IPHOTONUCR(
     *1)
      real*8 DUNIT,  RHOR
      integer*4 KMPI,  KMPO
      integer*2 MED,  IRAYLR,  IPHOTONUCR
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      common/x_options/eadl_relax,  mcdf_pe_xsections
      logical eadl_relax, mcdf_pe_xsections
      common/user_relax/ u_relax,ish_relax,iZ_relax
      real*8 u_relax
      integer*4 ish_relax, iZ_relax
      real*8 Ec,Pc,min_E,rnno,Evac,Ef,Ef1,Ef2,Ex,Ecc, cost,sint,cphi,sph
     *i
      integer*4 shell, shell_egs, iZ, iarg
      integer*4 irl,vacs(100),nvac,vac,new_state,iqf,np_save,new1,new2
      integer*4 sample_alias_histogram
      real*8 xphi,xphi2,yphi,yphi2,rhophi2
      shell = shell_eadl(iZ,shell_egs)
      IF (( shell .LT. 1 .OR. shell .GT. 3000 )) THEN
        return
      END IF
      irl = ir(np)
      Ec = ecut(irl) - rm
      Pc = pcut(irl)
      min_E = 0.001
      Evac = shell_be(shell)
      rfu_Z = shell_Z(shell)
      rfu_j0 = shell
      rfu_n0 = shell_num(shell)
      rfu_t0 = shell_type(shell)
      rfu_E0 = Evac
      IF ((shell_egs .GT. 4 .AND. .NOT.mcdf_pe_xsections)) THEN
        edep = Evac
        edep_local = Evac
        IARG=34
        IF ((IAUSFL(IARG+1).NE.0)) THEN
          CALL AUSGAB(IARG)
        END IF
        return
      END IF
      vac = shell
      Nvac = 0
      np_save = np
6511  CONTINUE
        IF (( Evac .LT. min_E .OR. relax_ntran(vac) .LT. 1 )) THEN
          edep = edep + Evac
          edep_local = Evac
          IARG=34
          IF ((IAUSFL(IARG+1).NE.0)) THEN
            CALL AUSGAB(IARG)
          END IF
          go to 6520
        END IF
        new_state = sample_alias_histogram(relax_ntran(vac), relax_prob(
     *  relax_first(vac)), relax_atbin(relax_first(vac)))
        IF (( new_state .LT. 0 )) THEN
          Ef = 0
          iqf = -1
          Ecc = Ec
        ELSE
          new_state = relax_state(relax_first(vac)+new_state-1)
          IF (( new_state .LE. 64 )) THEN
            iqf = 0
            new_state = new_state + vac - shell_num(vac)
            Ef = shell_be(new_state)
            Nvac = Nvac + 1
            vacs(Nvac) = new_state
            Ecc = Pc
          ELSE
            iqf = -1
            new1 = new_state/64
            new2 = new_state - 64*new1
            new1 = new1 + vac - shell_num(vac)
            new2 = new2 + vac - shell_num(vac)
            Ef1 = shell_be(new1)
            Ef2 = shell_be(new2)
            Nvac = Nvac + 1
            vacs(Nvac) = new1
            Nvac = Nvac + 1
            vacs(Nvac) = new2
            Ef = Ef1 + Ef2
            Ecc = Ec
          END IF
        END IF
        Ex = Evac - Ef
        edep_local = 0
        IF (( Ex .GT. Ecc )) THEN
          np = np + 1
          IF (( np .GT. 2 )) THEN
            write(i_log,'(/a)') '***************** Warning: '
            write(i_log,'(3(a,f10.6),a,i2)') 'Evac = ',Evac, ' Ef = ',Ef
     *      ,  ' min_E = ', min_E,' iq = ',iqf
            write(i_log,'(/a)') '***************** Error: '
            write(i_log,'(//,3a,/,2(a,i9),/,a)') ' In subroutine ','new_
     *relax', ' stack size exceeded! ',' $MXSTACK = ',2,' np = ',np, ' I
     *ncrease $MXSTACK and try again '
            write(i_log,'(/a)') '***************** Quiting now.'
            call exit(1)
          END IF
          iq(np) = iqf
          X(np)=X(np_save)
          Y(np)=Y(np_save)
          Z(np)=Z(np_save)
          IR(np)=IR(np_save)
          WT(np)=WT(np_save)
          DNEAR(np)=DNEAR(np_save)
          LATCH(np)=LATCH(np_save)
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          rnno = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          cost = 2*rnno-1
          sint = 1-cost*cost
          IF (( sint .GT. 0 )) THEN
            sint = sqrt(sint)
6531        CONTINUE
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              xphi = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              xphi = 2*xphi - 1
              xphi2 = xphi*xphi
              IF (( rng_seed .GT. 24 )) THEN
                call ranlux(rng_array)
                rng_seed = 1
              END IF
              yphi = rng_array(rng_seed)
              rng_seed = rng_seed + 1
              yphi2 = yphi*yphi
              rhophi2 = xphi2 + yphi2
              IF(rhophi2.LE.1)GO TO6532
            GO TO 6531
6532        CONTINUE
            rhophi2 = 1/rhophi2
            cphi = (xphi2 - yphi2)*rhophi2
            sphi = 2*xphi*yphi*rhophi2
            u(np) = sint*cphi
            v(np) = sint*sphi
            w(np) = cost
          ELSE
            u(np) = 0
            v(np) = 0
            w(np) = cost
          END IF
          rfu_j = vac
          rfu_n = shell_num(vac)
          rfu_t = shell_type(vac)
          rfu_E = shell_be(vac)
          IF (( iqf .EQ. 0 )) THEN
            e(np) = Ex
            IARG=25
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
          ELSE
            e(np) = Ex + rm
            IARG=27
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
          END IF
        ELSE
          edep = edep + Ex
          IF (( iqf .EQ. 0 )) THEN
            edep_local = Ex
            IARG=33
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
          ELSE
            edep_local = Ex
            IARG=34
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
          END IF
        END IF
6520    CONTINUE
        IF((Nvac .EQ. 0))GO TO6512
        vac = vacs(Nvac)
        Evac = shell_be(vac)
        Nvac = Nvac - 1
      GO TO 6511
6512  CONTINUE
      return
      end
      subroutine init_triplet
      implicit none
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      common/triplet_data/ a_triplet(250,1), b_triplet(250,1), dl_triple
     *t, dli_triplet, bli_triplet, log_4rm
      real*8 a_triplet,b_triplet,dl_triplet, dli_triplet, bli_triplet, l
     *og_4rm
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      real*8 energies(55), sig_pair(100,55), sig_triplet(100,55), f_trip
     *let(55), sigp(55), sigt(55), as(55), bs(55), cs(55), ds(55)
      character*128 triplet_data_file
      integer*4 want_triplet_unit, triplet_unit, triplet_out
      integer*4 i, iel, imed, lnblnk1, egs_get_unit, ntrip, iz1, izi, if
     *irst
      real*8 logE, f_new, f_old, spline
      IF((itriplet .EQ. 0))return
      DO 6541 i=1,len(triplet_data_file)
        triplet_data_file(i:i) = ' '
6541  CONTINUE
6542  CONTINUE
      triplet_data_file = hen_house(:lnblnk1(hen_house)) // 'data' // '/
     *' // 'triplet.data'
      want_triplet_unit = 63
      triplet_unit = egs_get_unit(want_triplet_unit)
      IF (( triplet_unit .LT. 1 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'init_triplet: failed to get a free Fortran I/O u
     *nit'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      open(triplet_unit,file=triplet_data_file,err=6550)
      write(i_log,'(a,$)') ' init_triplet: reading triplet data ... '
      read(triplet_unit,*) ntrip
      IF (( ntrip .GT. 55 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'Max. number of data points per element is ',55
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      read(triplet_unit,*,err=6560) (energies(i),i=1,ntrip)
      DO 6571 iel=1,100
        read(triplet_unit,*)
        read(triplet_unit,*,err=6560) (sig_pair(iel,i),i=1,ntrip)
        read(triplet_unit,*,err=6560) (sig_triplet(iel,i),i=1,ntrip)
6571  CONTINUE
6572  CONTINUE
      write(i_log,*) 'OK'
      ifirst = 0
      DO 6581 i=1,ntrip
        IF((ifirst .EQ. 0 .AND. energies(i) .GT. 4.01*rm))ifirst = i
        energies(i) = log(energies(i))
6581  CONTINUE
6582  CONTINUE
      log_4rm = log(4*rm)
      energies(ifirst-1) = log_4rm
      dl_triplet = (energies(ntrip) - log_4rm)/250
      dli_triplet = 1/dl_triplet
      bli_triplet = 1 - log_4rm/dl_triplet
      DO 6591 imed=1,nmed
        write(i_log,'(a,i3,a,$)') '   Preparing triplet fraction data fo
     *r medium ',imed,' ... '
        iz1 = zelem(imed,1) + 0.1
        DO 6601 i=1,ntrip
          sigp(i) = pz(imed,1)*sig_pair(iz1,i)
          sigt(i) = pz(imed,1)*sig_triplet(iz1,i)
          DO 6611 iel=2,nne(imed)
            izi = zelem(imed,iel) + 0.1
            sigp(i) = sigp(i) + pz(imed,iel)*sig_pair(izi,i)
            sigt(i) = sigt(i) + pz(imed,iel)*sig_triplet(izi,i)
6611      CONTINUE
6612      CONTINUE
6601    CONTINUE
6602    CONTINUE
        DO 6621 i=ifirst,ntrip
          f_triplet(i-ifirst+2) = sigt(i)/(sigp(i) + sigt(i))
6621    CONTINUE
6622    CONTINUE
        f_triplet(1) = 0
        call set_spline(energies(ifirst-1),f_triplet,as,bs,cs,ds,ntrip-i
     *  first+2)
        logE = log_4rm
        f_old = 0
        DO 6631 i=1,250-1
          logE = logE + dl_triplet
          f_new = spline(logE,energies(ifirst-1),as,bs,cs,ds,ntrip-ifirs
     *    t+2)
          a_triplet(i,imed) = (f_new - f_old)*dli_triplet
          b_triplet(i,imed) = f_new - a_triplet(i,imed)*logE
          f_old = f_new
6631    CONTINUE
6632    CONTINUE
        write(i_log,*) 'OK'
6591  CONTINUE
6592  CONTINUE
      close(triplet_unit)
      return
6550  CONTINUE
      write(i_log,'(/a)') '***************** Error: '
      write(i_log,'(a,a)') ' init_triplet: failed to open the data file
     *', triplet_data_file(:lnblnk1(triplet_data_file))
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
6560  CONTINUE
      write(i_log,'(/a)') '***************** Error: '
      write(i_log,*) ' init_triplet: error while reading triplet data '
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
      return
      end
      SUBROUTINE EDGSET(NREGLO,NREGHI)
      implicit none
      COMMON/EDGE/binding_energies(30,100), interaction_prob(6,100), rel
     *axation_prob(39,100), edge_energies(16,100), edge_number(100), edg
     *e_a(16,100), edge_b(16,100), edge_c(16,100), edge_d(16,100), IEDGF
     *L(1),IPHTER(1)
      real*8 binding_energies,  interaction_prob,    relaxation_prob,  e
     *dge_energies,  edge_a,edge_b,edge_c,edge_d
      integer*2 IEDGFL,  IPHTER
      integer*4 edge_number
      common/x_options/eadl_relax,  mcdf_pe_xsections
      logical eadl_relax, mcdf_pe_xsections
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      integer NREGLO,NREGHI
      integer*4 i,j,k,jj,iz
      logical do_relax
      logical got_data
      save got_data
      data got_data/.false./
      IF((got_data))return
      write(i_log,'(a/,a)') 'Output from subroutine EDGSET:', '=========
     *====================='
      do_relax = .false.
      DO 6641 j=1,1
        IF (( iedgfl(j) .GT. 0 .AND. iedgfl(j) .LE. 100 )) THEN
          do_relax = .true.
          GO TO6642
        END IF
6641  CONTINUE
6642  CONTINUE
      IF (( .NOT.do_relax )) THEN
        IF ((eadl_relax)) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,'(a,/a)') 'You must turn ON atomic relaxations whe
     *n requesting', 'detailed atomic relaxation (eadl_relax=true)!'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        write(i_log,'(a/)') ' Atomic relaxations not requested! '
        return
      END IF
      write(i_log,'(a/)') ' Atomic relaxations requested! '
      write(i_log,'(a$)') ' Reading simplified photo-absorption data ...
     *..'
      got_data = .true.
      rewind(i_photo_relax)
      DO 6651 i=1,100
        IF ((eadl_relax)) THEN
          read(i_photo_relax,*)
        ELSE
          read(i_photo_relax,*) j,(binding_energies(k,i),k=1,6)
          DO 6661 k=1,6
            binding_energies(k,i) = binding_energies(k,i)*1e-6
6661      CONTINUE
6662      CONTINUE
        END IF
6651  CONTINUE
6652  CONTINUE
      read(i_photo_relax,*)
      DO 6671 i=1,100
        read(i_photo_relax,*) j,(interaction_prob(k,i),k=1,5)
        interaction_prob(6,i)=1.01
6671  CONTINUE
6672  CONTINUE
      write(i_log,'(a)') ' Done'
      write(i_log,'(/a$)') ' Reading simplified relaxation data .....'
      read(i_photo_relax,*)
      DO 6681 i=1,100
        read(i_photo_relax,*) j,(relaxation_prob(k,i),k=1,19)
6681  CONTINUE
6682  CONTINUE
      read(i_photo_relax,*)
      DO 6691 i=1,100
        read(i_photo_relax,*) j,(relaxation_prob(k,i),k=20,26)
6691  CONTINUE
6692  CONTINUE
      read(i_photo_relax,*)
      DO 6701 i=1,100
        read(i_photo_relax,*) j,(relaxation_prob(k,i),k=27,32)
6701  CONTINUE
6702  CONTINUE
      read(i_photo_relax,*)
      DO 6711 i=1,100
        read(i_photo_relax,*) j,(relaxation_prob(k,i),k=33,37)
6711  CONTINUE
6712  CONTINUE
      read(i_photo_relax,*)
      DO 6721 i=1,100
        read(i_photo_relax,*) j,relaxation_prob(38,i)
6721  CONTINUE
6722  CONTINUE
      write(i_log,'(a)') ' Done'
      write(i_log,'(/a$)') ' Reading parametrized XCOM photo cross secti
     *on data .....'
      rewind(i_photo_cs)
      DO 6731 i=1,100
        read(i_photo_cs,*) j,edge_number(i)
        DO 6741 j=1,edge_number(i)
          read(i_photo_cs,*) edge_a(j,i),edge_b(j,i),edge_c(j,i), edge_d
     *    (j,i),edge_energies(j,i)
6741    CONTINUE
6742    CONTINUE
6731  CONTINUE
6732  CONTINUE
      write(i_log,'(a)') ' Done'
      IF ((eadl_relax)) THEN
        call egs_init_relax
      END IF
      RETURN
      END
      SUBROUTINE PHOTON(IRCODE)
      implicit none
      integer*4 IRCODE
      COMMON/QDEBUG/QDEBUG
      LOGICAL QDEBUG
      COMMON/BOUNDS/ECUT(1),PCUT(1),VACDST
      real*8 ECUT,  PCUT,  VACDST
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/MISC/  DUNIT,KMPI,KMPO,RHOR(1),MED(1),IRAYLR(1),IPHOTONUCR(
     *1)
      real*8 DUNIT,  RHOR
      integer*4 KMPI,  KMPO
      integer*2 MED,  IRAYLR,  IPHOTONUCR
      COMMON/EPCONT/EDEP,EDEP_LOCAL,TSTEP,TUSTEP,USTEP,TVSTEP,VSTEP, RHO
     *F,EOLD,ENEW,EKE,ELKE,GLE,E_RANGE, x_final,y_final,z_final, u_final
     *,v_final,w_final, IDISC,IROLD,IRNEW,IAUSFL(35)
      DOUBLE PRECISION EDEP,  EDEP_LOCAL
      real*8 TSTEP,  TUSTEP,  USTEP,  VSTEP,  TVSTEP,  RHOF,  EOLD,  ENE
     *W,  EKE,  ELKE,  GLE,  E_RANGE, x_final,y_final,z_final,  u_final,
     *v_final,w_final
      integer*4 IDISC,  IROLD,  IRNEW,  IAUSFL
      COMMON/PHOTIN/ EBINDA(1), GE0(1),GE1(1), GMFP0(2000,1),GMFP1(2000,
     *1),GBR10(2000,1),GBR11(2000,1),GBR20(2000,1),GBR21(2000,1), RCO0(1
     *),RCO1(1), RSCT0(100,1),RSCT1(100,1), COHE0(2000,1),COHE1(2000,1),
     *  PHOTONUC0(2000,1),PHOTONUC1(2000,1), DPMFP, MPGEM(1,1), NGR(1)
      real*8 EBINDA,  GE0,GE1,  GMFP0,GMFP1,  GBR10,GBR11,  GBR20,GBR21,
     *  RCO0,RCO1,  RSCT0,RSCT1,  COHE0,COHE1,   PHOTONUC0,PHOTONUC1,  D
     *PMFP
      integer*4
     *                  MPGEM,
     *                          NGR
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/UPHIOT/THETA,SINTHE,COSTHE,SINPHI, COSPHI,PI,TWOPI,PI5D2
      real*8 THETA,  SINTHE,  COSTHE,  SINPHI,  COSPHI,  PI,TWOPI,PI5D2
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      common/egs_vr/ e_max_rr(1),  prob_RR,  nbr_split,  i_play_RR,
     * i_survived_RR,
     *     n_RR_warning,                                        i_do_rr(
     *1)
      real*8          e_max_rr,prob_RR
      integer*4       nbr_split,i_play_RR,i_survived_RR,n_RR_warning
      integer*2     i_do_rr
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      DOUBLE PRECISION PEIG
      real*8 EIG,  RNNO35,  GMFPR0,  GMFP,  COHFAC,  RNNO37,  XXX,  X2,
     * Q2,  CSQTHE,  REJF,  RNNORJ,  RNNO36,  GBR1,  GBR2,  T,   PHOTONU
     *CFAC,  RNNO39
      integer*4 IARG,  IDR,  IRL,  LGLE,  LXXX
      IRCODE=1
      PEIG=E(NP)
      EIG=PEIG
      IRL=IR(NP)
      medium = med(irl)
      IF ((EIG .LE. PCUT(IRL))) THEN
        GO TO 6750
      END IF
6760  CONTINUE
6761    CONTINUE
        IF ((WT(NP) .EQ. 0.0)) THEN
          go to 6770
        END IF
        GLE=LOG(EIG)
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        RNNO35 = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        IF ((RNNO35.EQ.0.0)) THEN
          RNNO35=1.E-30
        END IF
        DPMFP=-LOG(RNNO35)
        IROLD=IR(NP)
6780    CONTINUE
6781      CONTINUE
          IF ((MEDIUM.NE.0)) THEN
            LGLE=GE1(MEDIUM)*GLE+GE0(MEDIUM)
            GMFPR0=GMFP1(LGLE,MEDIUM)*GLE+GMFP0(LGLE,MEDIUM)
          END IF
6790      CONTINUE
6791        CONTINUE
            IF ((MEDIUM.EQ.0)) THEN
              TSTEP=VACDST
            ELSE
              RHOF=RHOR(IRL)/RHO(MEDIUM)
              GMFP=GMFPR0/RHOF
              IF ((IRAYLR(IRL).EQ.1)) THEN
                COHFAC=COHE1(LGLE,MEDIUM)*GLE+COHE0(LGLE,MEDIUM)
                GMFP=GMFP*COHFAC
              END IF
              IF ((IPHOTONUCR(IRL).EQ.1)) THEN
                PHOTONUCFAC=PHOTONUC1(LGLE,MEDIUM)*GLE+PHOTONUC0(LGLE,ME
     *          DIUM)
                GMFP=GMFP*PHOTONUCFAC
              END IF
              TSTEP=GMFP*DPMFP
            END IF
            IRNEW=IR(NP)
            IDISC=0
            USTEP=TSTEP
            TUSTEP=USTEP
            IF (( ustep .GT. dnear(np) .OR. wt(np) .LE. 0 )) THEN
              call howfar
            END IF
            IF ((IDISC.GT.0)) THEN
              GO TO 6770
            END IF
            VSTEP=USTEP
            TVSTEP=VSTEP
            EDEP=PZERO
            x_final = x(np) + u(np)*vstep
            y_final = y(np) + v(np)*vstep
            z_final = z(np) + w(np)*vstep
            IARG=0
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
            x(np) = x_final
            y(np) = y_final
            z(np) = z_final
            DNEAR(NP)=DNEAR(NP)-USTEP
            IF ((MEDIUM.NE.0)) THEN
              DPMFP=MAX(0.,DPMFP-USTEP/GMFP)
            END IF
            IROLD=IR(NP)
            MEDOLD=MEDIUM
            IF ((IRNEW.NE.IROLD)) THEN
              ir(np) = irnew
              irl = irnew
              medium = med(irl)
            END IF
            IARG=5
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
            IF ((EIG.LE.PCUT(IRL))) THEN
              GO TO 6750
            END IF
            IF((IDISC.LT.0))GO TO 6770
            IF((MEDIUM.NE.MEDOLD))GO TO 6792
            IF ((MEDIUM.NE.0.AND.DPMFP.LE.1.E-8)) THEN
              GO TO 6782
            END IF
          GO TO 6791
6792      CONTINUE
        GO TO 6781
6782    CONTINUE
        IF ((IRAYLR(IRL).EQ.1)) THEN
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          RNNO37 = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          IF ((RNNO37.LE.(1.0-COHFAC))) THEN
            IARG=23
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
            NPold = NP
            call egs_rayleigh_sampling(MEDIUM,E(NP),GLE,LGLE,COSTHE,SINT
     *      HE)
            CALL UPHI(2,1)
            IARG=24
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
            GOTO 6760
          END IF
        END IF
        IF ((IPHOTONUCR(IRL).EQ.1)) THEN
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          RNNO39 = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          IF ((RNNO39.LE.(1.0-PHOTONUCFAC))) THEN
            IARG=29
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
            call PHOTONUC
            IARG=30
            IF ((IAUSFL(IARG+1).NE.0)) THEN
              CALL AUSGAB(IARG)
            END IF
            GOTO 6760
          END IF
        END IF
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        RNNO36 = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        GBR1=GBR11(LGLE,MEDIUM)*GLE+GBR10(LGLE,MEDIUM)
        IF (((RNNO36.LE.GBR1).AND.(E(NP).GT.RMT2) )) THEN
          IARG=15
          IF ((IAUSFL(IARG+1).NE.0)) THEN
            CALL AUSGAB(IARG)
          END IF
          CALL PAIR
          IARG=16
          IF ((IAUSFL(IARG+1).NE.0)) THEN
            CALL AUSGAB(IARG)
          END IF
          IF (( iq(np) .NE. 0 )) THEN
            GO TO 6762
          ELSE
            goto 6800
          END IF
        END IF
        GBR2=GBR21(LGLE,MEDIUM)*GLE+GBR20(LGLE,MEDIUM)
        IF ((RNNO36.LT.GBR2)) THEN
          IARG=17
          IF ((IAUSFL(IARG+1).NE.0)) THEN
            CALL AUSGAB(IARG)
          END IF
          CALL COMPT
          IARG=18
          IF ((IAUSFL(IARG+1).NE.0)) THEN
            CALL AUSGAB(IARG)
          END IF
          IF((IQ(NP).NE.0))GO TO 6762
        ELSE
          IARG=19
          IF ((IAUSFL(IARG+1).NE.0)) THEN
            CALL AUSGAB(IARG)
          END IF
          CALL PHOTO
          IF ((NP .EQ. 0 .OR. NP .LT. NPOLD )) THEN
            RETURN
          END IF
          IARG=20
          IF ((IAUSFL(IARG+1).NE.0)) THEN
            CALL AUSGAB(IARG)
          END IF
          IF((IQ(NP) .NE. 0))GO TO 6762
        END IF
6800    PEIG=E(NP)
        EIG=PEIG
        IF((EIG.LT.PCUT(IRL)))GO TO 6750
      GO TO 6761
6762  CONTINUE
      RETURN
6750  IF (( medium .GT. 0 )) THEN
        IF ((EIG.GT.AP(MEDIUM))) THEN
          IDR=1
        ELSE
          IDR=2
        END IF
      ELSE
        IDR=1
      END IF
      EDEP=PEIG
      IARG=IDR
      IF ((IAUSFL(IARG+1).NE.0)) THEN
        CALL AUSGAB(IARG)
      END IF
      IRCODE=2
      NP=NP-1
      RETURN
6770  EDEP=PEIG
      IARG=3
      IF ((IAUSFL(IARG+1).NE.0)) THEN
        CALL AUSGAB(IARG)
      END IF
      IRCODE=2
      NP=NP-1
      RETURN
      END
      SUBROUTINE SHOWER(IQI,EI,XI,YI,ZI,UI,VI,WI,IRI,WTI)
      implicit none
      COMMON/QDEBUG/QDEBUG
      LOGICAL QDEBUG
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/UPHIOT/THETA,SINTHE,COSTHE,SINPHI, COSPHI,PI,TWOPI,PI5D2
      real*8 THETA,  SINTHE,  COSTHE,  SINPHI,  COSPHI,  PI,TWOPI,PI5D2
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      real*8 EI,  XI,YI,ZI, UI,VI,WI, WTI
      integer*4 IQI,  IRI
      DOUBLE PRECISION DEG,  DPGL,  DEI,  DPI,  DCSTH,  DCOSTH,  PI0MSQ
      real*8 DNEARI,  CSTH
      integer*4 IRCODE
      DATA PI0MSQ/1.8215416D4/
      NP=1
      NPold = NP
      DNEARI=0.0
      IQ(1)=IQI
      E(1)=EI
      U(1)=UI
      V(1)=VI
      W(1)=WI
      X(1)=XI
      Y(1)=YI
      Z(1)=ZI
      IR(1)=IRI
      WT(1)=WTI
      DNEAR(1)=DNEARI
      LATCH(1)=LATCHI
      IF ((IQI .EQ. 2)) THEN
        IF ((EI**2 .LE. PI0MSQ)) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,'(//a/,a,g15.5,a)') ' Stopped in subroutine SHOWER
     *---PI-ZERO option invoked', ' but the total energy was too small (
     *EI=',EI,' MeV)'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        CSTH = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        DCSTH=CSTH
        DEI=EI
        DPI=DSQRT(DEI*DEI-PI0MSQ)
        DEG=DEI+DPI*DCSTH
        DPGL=DPI+DEI*DCSTH
        DCOSTH=DPGL/DEG
        COSTHE=DCOSTH
        SINTHE=DSQRT(1.D0-DCOSTH*DCOSTH)
        IQ(1)=0
        E(1)=DEG/2.
        CALL UPHI(2,1)
        NP=2
        DEG=DEI-DPI*DCSTH
        DPGL=DPI-DEI*DCSTH
        DCOSTH=DPGL/DEG
        COSTHE=DCOSTH
        SINTHE=-DSQRT(1.D0-DCOSTH*DCOSTH)
        IQ(2)=0
        E(2)=DEG/2.
        CALL UPHI(3,2)
      END IF
6811  CONTINUE
        IF((np .LE. 0))GO TO6812
        IF (( iq(np) .EQ. 0 )) THEN
          call photon(ircode)
        ELSE
          call electr(ircode)
        END IF
      GO TO 6811
6812  CONTINUE
      RETURN
      END
      SUBROUTINE UPHI(IENTRY,LVL)
      implicit none
      COMMON/QDEBUG/QDEBUG
      LOGICAL QDEBUG
      COMMON/EPCONT/EDEP,EDEP_LOCAL,TSTEP,TUSTEP,USTEP,TVSTEP,VSTEP, RHO
     *F,EOLD,ENEW,EKE,ELKE,GLE,E_RANGE, x_final,y_final,z_final, u_final
     *,v_final,w_final, IDISC,IROLD,IRNEW,IAUSFL(35)
      DOUBLE PRECISION EDEP,  EDEP_LOCAL
      real*8 TSTEP,  TUSTEP,  USTEP,  VSTEP,  TVSTEP,  RHOF,  EOLD,  ENE
     *W,  EKE,  ELKE,  GLE,  E_RANGE, x_final,y_final,z_final,  u_final,
     *v_final,w_final
      integer*4 IDISC,  IROLD,  IRNEW,  IAUSFL
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/UPHIIN/SINC0,SINC1,SIN0(1002),SIN1(1002)
      real*8 SINC0,SINC1,SIN0,SIN1
      COMMON/UPHIOT/THETA,SINTHE,COSTHE,SINPHI, COSPHI,PI,TWOPI,PI5D2
      real*8 THETA,  SINTHE,  COSTHE,  SINPHI,  COSPHI,  PI,TWOPI,PI5D2
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      integer IENTRY,LVL
      real*8 CTHET,  RNNO38,  PHI,  CPHI,  A,B,C,  SINPS2,  SINPSI,  US,
     *VS,  SINDEL,COSDEL
      integer*4 IARG,  LPHI,LTHETA,LCTHET,LCPHI
      real*8 xphi,xphi2,yphi,yphi2,rhophi2
      save CTHET,PHI,CPHI,A,B,C,SINPS2,SINPSI,US,VS,SINDEL,COSDEL
      IARG=21
      IF ((IAUSFL(IARG+1).NE.0)) THEN
        CALL AUSGAB(IARG)
      END IF
      GO TO (6820,6830,6840),IENTRY
      GO TO 6850
6820  CONTINUE
      SINTHE=sin(THETA)
      CTHET=PI5D2-THETA
      COSTHE=sin(CTHET)
6830  CONTINUE
6861  CONTINUE
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        xphi = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        xphi = 2*xphi - 1
        xphi2 = xphi*xphi
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        yphi = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        yphi2 = yphi*yphi
        rhophi2 = xphi2 + yphi2
        IF(rhophi2.LE.1)GO TO6862
      GO TO 6861
6862  CONTINUE
      rhophi2 = 1/rhophi2
      cosphi = (xphi2 - yphi2)*rhophi2
      sinphi = 2*xphi*yphi*rhophi2
6840  GO TO (6870,6880,6890),LVL
      GO TO 6850
6870  A=U(NP)
      B=V(NP)
      C=W(NP)
      GO TO 6900
6890  A=U(NP-1)
      B=V(NP-1)
      C=W(NP-1)
6880  X(NP)=X(NP-1)
      Y(NP)=Y(NP-1)
      Z(NP)=Z(NP-1)
      IR(NP)=IR(NP-1)
      WT(NP)=WT(NP-1)
      DNEAR(NP)=DNEAR(NP-1)
      LATCH(NP)=LATCH(NP-1)
6900  SINPS2=A*A+B*B
      IF ((SINPS2.LT.1.0E-20)) THEN
        U(NP)=SINTHE*COSPHI
        V(NP)=SINTHE*SINPHI
        W(NP)=C*COSTHE
      ELSE
        SINPSI=SQRT(SINPS2)
        US=SINTHE*COSPHI
        VS=SINTHE*SINPHI
        SINDEL=B/SINPSI
        COSDEL=A/SINPSI
        U(NP)=C*COSDEL*US-SINDEL*VS+A*COSTHE
        V(NP)=C*SINDEL*US+COSDEL*VS+B*COSTHE
        W(NP)=-SINPSI*US+C*COSTHE
      END IF
      IARG=22
      IF ((IAUSFL(IARG+1).NE.0)) THEN
        CALL AUSGAB(IARG)
      END IF
      RETURN
6850  write(i_log,'(/a)') '***************** Error: '
      write(i_log,'(a,2i6)') ' STOPPED IN UPHI WITH IENTRY,LVL=',IENTRY,
     *LVL
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
      END
      subroutine init_nist_brems
      implicit none
      real*8 energy_array(57),x_array(54), cs_array(57,54,100)
      real*8 xi_array(54)
      real*8 x_gauss(64),w_gauss(64)
      integer*4 nmix,kmix,i,n,k,j,ii
      integer*4 ngauss,i_gauss
      integer*4 lnblnk1,egs_get_unit
      integer*4 ifirst,ilast,nener,neke,leil
      real*8 cs(57,54),ee(57),ele(57)
      real*8 csx(54),afx(54),bfx(54),cfx(54),dfx(54)
      real*8 cse(57),afe(57),bfe(57),cfe(57),dfe(57)
      real*8 Z,sumA
      real*8 emin,xi,res,spline,eil,ei,beta2,aux,sigb,sigt,ebr1,ebr2
      real*8 sigee,sigep,sige,si_esig,si1_esig,si_ebr1,si1_ebr1,ededx, s
     *ig_bhabha,si_psig,si1_psig,si_pbr1,si1_pbr1,si_pbr2,si1_pbr2
      integer*4 iz
      real*8 ple,qle,x,f,error,max_error,x_max_error,f_max_error
      integer*4 ndat,k_max_error
      character tmp_string*512, tmp1_string*512
      integer itmp
      real*8 amu
      parameter (amu = 1660.5655)
      logical ex,is_opened
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      COMMON/ELECIN/ esig_e(1),psig_e(1), esige_max, psige_max, range_ep
     *(0:1,500,1), E_array(500,1), etae_ms0(500,1),etae_ms1(500,1),etap_
     *ms0(500,1),etap_ms1(500,1),q1ce_ms0(500,1),q1ce_ms1(500,1),q1cp_ms
     *0(500,1),q1cp_ms1(500,1),q2ce_ms0(500,1),q2ce_ms1(500,1),q2cp_ms0(
     *500,1),q2cp_ms1(500,1),blcce0(500,1),blcce1(500,1), EKE0(1),EKE1(1
     *), XR0(1),TEFF0(1),BLCC(1),XCC(1), ESIG0(500,1),ESIG1(500,1),PSIG0
     *(500,1),PSIG1(500,1),EDEDX0(500,1),EDEDX1(500,1),PDEDX0(500,1),PDE
     *DX1(500,1),EBR10(500,1),EBR11(500,1),PBR10(500,1),PBR11(500,1),PBR
     *20(500,1),PBR21(500,1),TMXS0(500,1),TMXS1(500,1), expeke1(1), IUNR
     *ST(1),EPSTFL(1),IAPRIM(1), sig_ismonotone(0:1,1)
      real*8 esig_e,   psig_e,   esige_max,  psige_max,  range_ep,  E_ar
     *ray,  etae_ms0,etae_ms1,  etap_ms0,etap_ms1,  q1ce_ms0,q1ce_ms1,
     *q1cp_ms0,q1cp_ms1,  q2ce_ms0,q2ce_ms1,  q2cp_ms0,q2cp_ms1,  blcce0
     *,blcce1,   expeke1,  EKE0,EKE1, XR0,  TEFF0,  BLCC,  XCC,  ESIG0,E
     *SIG1,  PSIG0,PSIG1,  EDEDX0,EDEDX1,  PDEDX0,PDEDX1,  EBR10,EBR11,
     * PBR10,PBR11,  PBR20,PBR21,  TMXS0,TMXS1
      integer*4 IUNRST,  EPSTFL,  IAPRIM
      logical sig_ismonotone
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common/nist_brems/ nb_fdata(0:50,100,1), nb_xdata(0:50,100,1), nb_
     *wdata(50,100,1), nb_idata(50,100,1), nb_emin(1),nb_emax(1), nb_lem
     *in(1),nb_lemax(1), nb_dle(1),nb_dlei(1), log_ap(1)
      real*8 nb_fdata,nb_xdata,nb_wdata,nb_emin,nb_emax,nb_lemin,nb_lema
     *x, nb_dle,nb_dlei,log_ap
      integer*4 nb_idata
      common/spin_data/ spin_rej(1,0:1,0: 31,0:15,0:31), espin_min,espin
     *_max,espml,b2spin_min,b2spin_max, dbeta2,dbeta2i,dlener,dleneri,dq
     *q1,dqq1i, fool_intel_optimizer
      real*4 spin_rej,espin_min,espin_max,espml,b2spin_min,b2spin_max, d
     *beta2,dbeta2i,dlener,dleneri,dqq1,dqq1i
      logical fool_intel_optimizer
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      DO 6911 i=1,len(tmp_string)
        tmp_string(i:i) = ' '
6911  CONTINUE
6912  CONTINUE
      tmp_string = hen_house(:lnblnk1(hen_house)) // 'data' // '/'
      IF (( ibr_nist .EQ. 1 )) THEN
        DO 6921 i=1,len(tmp1_string)
          tmp1_string(i:i) = ' '
6921    CONTINUE
6922    CONTINUE
        tmp1_string = tmp_string(:lnblnk1(tmp_string)) // 'nist_brems.da
     *ta'
        inquire(file=tmp1_string,exist=ex,opened=is_opened,number=itmp)
        IF (( .NOT.ex )) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,*) 'EGSnrc data file ','nist_brems.data',' does no
     *t exist'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        IF (( .NOT.is_opened )) THEN
          i_nist_data=egs_get_unit(i_nist_data)
          IF ((i_nist_data.LT.0)) THEN
            write(i_log,'(/a)') '***************** Error: '
            write(i_log,*) 'failed to get a free Fortran I/O unit for da
     *ta file ', tmp1_string(:lnblnk1(tmp1_string))
            write(i_log,'(/a)') '***************** Quiting now.'
            call exit(1)
          END IF
          open(i_nist_data,file=tmp1_string,status='old',err=2650)
        ELSE
          i_nist_data = itmp
        END IF
      ELSE IF((ibr_nist .EQ. 2)) THEN
        DO 6931 i=1,len(tmp1_string)
          tmp1_string(i:i) = ' '
6931    CONTINUE
6932    CONTINUE
        tmp1_string = tmp_string(:lnblnk1(tmp_string)) // 'nrc_brems.dat
     *a'
        inquire(file=tmp1_string,exist=ex,opened=is_opened,number=itmp)
        IF (( .NOT.ex )) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,*) 'EGSnrc data file ','nrc_brems.data',' does not
     * exist'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        IF (( .NOT.is_opened )) THEN
          i_nist_data=egs_get_unit(i_nist_data)
          IF ((i_nist_data.LT.0)) THEN
            write(i_log,'(/a)') '***************** Error: '
            write(i_log,*) 'failed to get a free Fortran I/O unit for da
     *ta file ', tmp1_string(:lnblnk1(tmp1_string))
            write(i_log,'(/a)') '***************** Quiting now.'
            call exit(1)
          END IF
          open(i_nist_data,file=tmp1_string,status='old',err=2650)
        ELSE
          i_nist_data = itmp
        END IF
      ELSE
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) ' init_nist_brems: unknown value of ibr_nist!
     *                  ibr_nist = ', ibr_nist
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      rewind(i_nist_data)
      read(i_nist_data,*)
      read(i_nist_data,*) nmix,kmix
      IF ((kmix .GT. 54)) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) ' init_nist_brems: to many k values in data file!
     *'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      IF ((nmix .GT. 57)) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) ' init_nist_brems: to many T values in data file!
     *'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      read(i_nist_data,*) (energy_array(n),n=1,nmix)
      DO 6941 n=1,nmix
        energy_array(n) = 1.0*energy_array(n)
6941  CONTINUE
6942  CONTINUE
      read(i_nist_data,*) (x_array(k),k=1,kmix)
      read(i_nist_data,*)
      DO 6951 i=1,100
        read(i_nist_data,*) ((cs_array(n,k,i),n=1,nmix),k=1,kmix)
6951  CONTINUE
6952  CONTINUE
      close(i_nist_data)
      DO 6961 k=1,kmix
        xi_array(k)=Log(1-x_array(k)+1e-6)
        IF (( fool_intel_optimizer )) THEN
          write(i_log,*) 'xi_array(k): ',xi_array(k)
        END IF
6961  CONTINUE
6962  CONTINUE
      ngauss = 64
      call gauss_legendre(0d0,1d0,x_gauss,w_gauss,ngauss)
      write(i_log,*) ' '
      IF ((ibr_nist .EQ. 1)) THEN
        write(i_log,*) 'Using NIST brems cross sections! '
      ELSE IF((ibr_nist .EQ. 2)) THEN
        write(i_log,*) 'Using NRC brems cross sections! '
      END IF
      write(i_log,*) ' '
      DO 6971 medium=1,nmed
        log_ap(medium) = log(ap(medium))
        write(i_log,*) ' Initializing brems data for medium ',medium,'..
     *.'
        emin = max(ae(medium) - rm, ap(medium))
        DO 6981 i=1,nmix
          IF((energy_array(i) .GE. emin))GO TO6982
6981    CONTINUE
6982    CONTINUE
        ifirst = i
        DO 6991 i=nmix,1,-1
          IF((energy_array(i) .LT. ue(medium) - rm))GO TO6992
6991    CONTINUE
6992    CONTINUE
        ilast = i+1
        IF (( ifirst .LT. 1 .OR. ilast .GT. nmix )) THEN
          write(i_log,*) ' init_nist_brems: data available only for '
          write(i_log,*) energy_array(1),' <= E <= ',energy_array(nmix)
          write(i_log,*) ' will use spline interpolations to get cross '
          write(i_log,*) ' sections beyond the available data but this m
     *ay'
          write(i_log,*) ' produce nonsense!'
          IF((ifirst .LT. 1))ifirst=1
          IF((ilast .GT. nmix))ilast = nmix
        END IF
        DO 7001 i=ifirst,ilast
          ii = i+1 - ifirst
          ee(ii) = energy_array(i)
          ele(ii) = log(ee(ii))
          sumA = 0
          DO 7011 j=1,NNE(medium)
            sumA = sumA + pz(medium,j)*wa(medium,j)
7011      CONTINUE
7012      CONTINUE
          sumA = sumA*amu
          DO 7021 k=1,kmix
            cs(ii,k) = 0
            DO 7031 j=1,NNE(medium)
              Z = zelem(medium,j)
              iz = int(Z+0.1)
              Z = Z*Z/sumA
              cs(ii,k) = cs(ii,k) + pz(medium,j)*Z*cs_array(i,k,iz)
7031        CONTINUE
7032        CONTINUE
            csx(k) = Log(cs(ii,k))
7021      CONTINUE
7022      CONTINUE
          call set_spline(xi_array,csx,afx,bfx,cfx,dfx,kmix)
          cse(ii) = 0
          aux = Log(ee(ii)/ap(medium))
          DO 7041 i_gauss=1,ngauss
            xi = log(1 - ap(medium)/ee(ii)*exp(x_gauss(i_gauss)*aux)+1e-
     *      6)
            res = spline(xi,xi_array,afx,bfx,cfx,dfx,kmix)
            cse(ii) = cse(ii) + w_gauss(i_gauss)*exp(res)
7041      CONTINUE
7042      CONTINUE
7001    CONTINUE
7002    CONTINUE
        nener = ilast - ifirst + 1
        call set_spline(ele,cse,afe,bfe,cfe,dfe,nener)
        neke = meke(medium)
        sigee = 1E-15
        sigep = 1E-15
        DO 7051 i=1,neke
          eil = (float(i) - eke0(medium))/eke1(medium)
          ei = exp(eil)
          leil = i
          beta2 = ei*(ei+2*rm)/(ei+rm)**2
          IF (( ei .LE. ap(medium) )) THEN
            sigb = 1e-30
          ELSE
            sigb = spline(eil,ele,afe,bfe,cfe,dfe,nener)
            sigb = sigb*log(ei/ap(medium))/beta2*rho(medium)
          END IF
          sigt=esig1(Leil,MEDIUM)*eil+esig0(Leil,MEDIUM)
          ebr1=ebr11(Leil,MEDIUM)*eil+ebr10(Leil,MEDIUM)
          IF((sigt .LT. 0))sigt = 0
          IF((ebr1 .GT. 1))ebr1 = 1
          IF((ebr1 .LT. 0))ebr1 = 0
          IF (( i .GT. 1 )) THEN
            si_esig = si1_esig
            si_ebr1 = si1_ebr1
            si1_esig = sigt*(1 - ebr1) + sigb
            si1_ebr1 = sigb/si1_esig
            esig1(i-1,medium) = (si1_esig - si_esig)*eke1(medium)
            esig0(i-1,medium) = si1_esig - esig1(i-1,medium)*eil
            ebr11(i-1,medium) = (si1_ebr1 - si_ebr1)*eke1(medium)
            ebr10(i-1,medium) = si1_ebr1 - ebr11(i-1,medium)*eil
          ELSE
            si1_esig = sigt*(1 - ebr1) + sigb
            si1_ebr1 = sigb/si1_esig
          END IF
          sigt=psig1(Leil,MEDIUM)*eil+psig0(Leil,MEDIUM)
          ebr1=pbr11(Leil,MEDIUM)*eil+pbr10(Leil,MEDIUM)
          ebr2=pbr21(Leil,MEDIUM)*eil+pbr20(Leil,MEDIUM)
          IF((sigt .LT. 0))sigt = 0
          IF((ebr1 .GT. 1))ebr1 = 1
          IF((ebr1 .LT. 0))ebr1 = 0
          IF((ebr2 .GT. 1))ebr2 = 1
          IF((ebr2 .LT. 0))ebr2 = 0
          sig_bhabha = sigt*(ebr2 - ebr1)
          IF((sig_bhabha .LT. 0))sig_bhabha = 0
          IF (( i .GT. 1 )) THEN
            si_psig = si1_psig
            si_pbr1 = si1_pbr1
            si_pbr2 = si1_pbr2
            si1_psig = sigt*(1 - ebr1) + sigb
            si1_pbr1 = sigb/si1_psig
            si1_pbr2 = (sigb + sig_bhabha)/si1_psig
            psig1(i-1,medium) = (si1_psig - si_psig)*eke1(medium)
            psig0(i-1,medium) = si1_psig - psig1(i-1,medium)*eil
            pbr11(i-1,medium) = (si1_pbr1 - si_pbr1)*eke1(medium)
            pbr10(i-1,medium) = si1_pbr1 - pbr11(i-1,medium)*eil
            pbr21(i-1,medium) = (si1_pbr2 - si_pbr2)*eke1(medium)
            pbr20(i-1,medium) = si1_pbr2 - pbr21(i-1,medium)*eil
          ELSE
            si1_psig = sigt*(1 - ebr1) + sigb
            si1_pbr1 = sigb/si1_psig
            si1_pbr2 = (sigb + sig_bhabha)/si1_psig
          END IF
          ededx=ededx1(Leil,MEDIUM)*eil+ededx0(Leil,MEDIUM)
          sige = si1_esig/ededx
          IF((sige .GT. sigee))sigee = sige
          ededx=pdedx1(Leil,MEDIUM)*eil+pdedx0(Leil,MEDIUM)
          sige = si1_psig/ededx
          IF((sige .GT. sigep))sigep = sige
7051    CONTINUE
7052    CONTINUE
        esig1(neke,medium) = esig1(neke-1,medium)
        esig0(neke,medium) = esig0(neke-1,medium)
        ebr11(neke,medium) = ebr11(neke-1,medium)
        ebr10(neke,medium) = ebr10(neke-1,medium)
        psig1(neke,medium) = psig1(neke-1,medium)
        psig0(neke,medium) = psig0(neke-1,medium)
        pbr11(neke,medium) = pbr11(neke-1,medium)
        pbr10(neke,medium) = pbr10(neke-1,medium)
        pbr21(neke,medium) = pbr21(neke-1,medium)
        pbr20(neke,medium) = pbr20(neke-1,medium)
        write(i_log,*) ' Max. new cross sections per energy loss: ',sige
     *  e,sigep
        esig_e(medium) = sigee
        psig_e(medium) = sigep
        IF((sigee .GT. esige_max))esige_max = sigee
        IF((sigep .GT. psige_max))psige_max = sigep
        nb_emin(medium) = energy_array(ifirst)
        IF (( nb_emin(medium) .LE. ap(medium) )) THEN
          nb_emin(medium) = energy_array(ifirst+1)
        END IF
        nb_emax(medium) = energy_array(ilast)
        nb_lemin(medium) = log(nb_emin(medium))
        nb_lemax(medium) = log(nb_emax(medium))
        nb_dle(medium) = (nb_lemax(medium) - nb_lemin(medium))/(100-1)
        nb_dlei(medium) = 1/nb_dle(medium)
        eil = nb_lemin(medium) - nb_dle(medium)
        DO 7061 i=1,100
          eil = eil + nb_dle(medium)
          ei = exp(eil)
          DO 7071 ii=1,nener
            IF((ei .LT. ee(ii)))GO TO7072
7071      CONTINUE
7072      CONTINUE
          ii = ii-1
          IF((ii .LT. 1))ii = 1
          IF((ii .GT. nener-1))ii = nener-1
          ple = (eil - ele(ii))/(ele(ii+1)-ele(ii))
          qle = 1 - ple
          DO 7081 k=1,kmix
            csx(k) = log(qle*cs(ii,k) + ple*cs(ii+1,k))
7081      CONTINUE
7082      CONTINUE
          call set_spline(xi_array,csx,afx,bfx,cfx,dfx,kmix)
          x = ap(medium)/ei
          aux = -log(x)
          xi = log(1 - x+1e-6)
          res = spline(xi,xi_array,afx,bfx,cfx,dfx,kmix)
          nb_xdata(0,i,medium) = 0
          nb_fdata(0,i,medium) = exp(res)
          DO 7091 k=1,kmix
            IF((x_array(k) .GT. x))GO TO7092
7091      CONTINUE
7092      CONTINUE
          IF((k .GT. kmix))k = kmix
          ndat = 0
          DO 7101 j=k+1,kmix-1
            ndat = ndat+1
            nb_xdata(ndat,i,medium) = log(x_array(j)/x)/aux
            nb_fdata(ndat,i,medium) = exp(csx(j))
            IF (( fool_intel_optimizer )) THEN
              write(i_log,*) 'nb_xdata(ndat,i,medium): ', nb_xdata(ndat,
     *        i,medium)
            END IF
7101      CONTINUE
7102      CONTINUE
          ndat = ndat+1
          nb_xdata(ndat,i,medium) = 1
          nb_fdata(ndat,i,medium) = exp(csx(kmix))
          IF((ndat .GE. 50))goto 7110
7121      CONTINUE
            x_max_error = 0
            f_max_error = 0
            k_max_error = 0
            max_error = 0
            DO 7131 k=0,ndat-1
              x = 0.5*(nb_xdata(k,i,medium) + nb_xdata(k+1,i,medium))
              f = 0.5*(nb_fdata(k,i,medium) + nb_fdata(k+1,i,medium))
              xi = log(1 - ap(medium)/ei*exp(x*aux)+1e-6)
              res = spline(xi,xi_array,afx,bfx,cfx,dfx,kmix)
              res = exp(res)
              error = abs(1-f/res)
              IF (( error .GT. max_error )) THEN
                x_max_error = x
                f_max_error = res
                max_error = error
                k_max_error = k
              END IF
7131        CONTINUE
7132        CONTINUE
            ndat = ndat+1
            DO 7141 k=ndat,k_max_error+2,-1
              nb_xdata(k,i,medium) = nb_xdata(k-1,i,medium)
              nb_fdata(k,i,medium) = nb_fdata(k-1,i,medium)
7141        CONTINUE
7142        CONTINUE
            nb_xdata(k_max_error+1,i,medium) = x_max_error
            nb_fdata(k_max_error+1,i,medium) = f_max_error
            IF(((ndat .EQ. 50)))GO TO7122
          GO TO 7121
7122      CONTINUE
7110      call prepare_alias_table(50,nb_xdata(0,i,medium), nb_fdata(0,i
     *    ,medium),nb_wdata(1,i,medium),nb_idata(1,i,medium))
7061    CONTINUE
7062    CONTINUE
6971  CONTINUE
6972  CONTINUE
      write(i_log,*) ' '
      write(i_log,*) ' '
      return
2650  write(i_log,'(/a)') '***************** Error: '
      write(i_log,*) 'failed to open EGSnrc data file ',tmp1_string(:lnb
     *lnk1(tmp1_string))
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
      return
      end
      subroutine init_nrc_pair
      implicit none
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      COMMON/ELECIN/ esig_e(1),psig_e(1), esige_max, psige_max, range_ep
     *(0:1,500,1), E_array(500,1), etae_ms0(500,1),etae_ms1(500,1),etap_
     *ms0(500,1),etap_ms1(500,1),q1ce_ms0(500,1),q1ce_ms1(500,1),q1cp_ms
     *0(500,1),q1cp_ms1(500,1),q2ce_ms0(500,1),q2ce_ms1(500,1),q2cp_ms0(
     *500,1),q2cp_ms1(500,1),blcce0(500,1),blcce1(500,1), EKE0(1),EKE1(1
     *), XR0(1),TEFF0(1),BLCC(1),XCC(1), ESIG0(500,1),ESIG1(500,1),PSIG0
     *(500,1),PSIG1(500,1),EDEDX0(500,1),EDEDX1(500,1),PDEDX0(500,1),PDE
     *DX1(500,1),EBR10(500,1),EBR11(500,1),PBR10(500,1),PBR11(500,1),PBR
     *20(500,1),PBR21(500,1),TMXS0(500,1),TMXS1(500,1), expeke1(1), IUNR
     *ST(1),EPSTFL(1),IAPRIM(1), sig_ismonotone(0:1,1)
      real*8 esig_e,   psig_e,   esige_max,  psige_max,  range_ep,  E_ar
     *ray,  etae_ms0,etae_ms1,  etap_ms0,etap_ms1,  q1ce_ms0,q1ce_ms1,
     *q1cp_ms0,q1cp_ms1,  q2ce_ms0,q2ce_ms1,  q2cp_ms0,q2cp_ms1,  blcce0
     *,blcce1,   expeke1,  EKE0,EKE1, XR0,  TEFF0,  BLCC,  XCC,  ESIG0,E
     *SIG1,  PSIG0,PSIG1,  EDEDX0,EDEDX1,  PDEDX0,PDEDX1,  EBR10,EBR11,
     * PBR10,PBR11,  PBR20,PBR21,  TMXS0,TMXS1
      integer*4 IUNRST,  EPSTFL,  IAPRIM
      logical sig_ismonotone
      common/nrc_pair/ nrcp_fdata(65,84,1), nrcp_wdata(65,84,1), nrcp_id
     *ata(65,84,1), nrcp_xdata(65), nrcp_emin, nrcp_emax, nrcp_dle, nrcp
     *_dlei
      real*8 nrcp_fdata,nrcp_wdata,nrcp_xdata, nrcp_emin, nrcp_emax, nrc
     *p_dle, nrcp_dlei
      integer*4 nrcp_idata
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      character nrcp_file*256, endianess*4
      integer egs_get_unit
      integer*4 nrcp_unit, want_nrcp_unit, rec_length
      integer*4 i, lnblnk1
      real*8 tmp, ddx, xx, Z
      real*4 emin, emax
      integer*4 ne, nb, ix, ie, irec, i_ele, nbb, iz
      character endian, cdum( 243)
      logical swap
      real*4 tmp_4, tarray(65)
      integer*4 itmp_4
      character c_4(4), ic_4(4)
      equivalence (tmp_4,c_4), (itmp_4, ic_4)
      DO 7151 i=1,len(nrcp_file)
        nrcp_file(i:i) = ' '
7151  CONTINUE
7152  CONTINUE
      nrcp_file = hen_house(:lnblnk1(hen_house)) // 'data' // '/' // 'pa
     *ir_nrc1.data'
      want_nrcp_unit = 62
      nrcp_unit = egs_get_unit(want_nrcp_unit)
      IF (( nrcp_unit .LT. 1 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'init_nrc_pair: failed to get a free fortran unit
     *'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      rec_length = 65*4
      open(nrcp_unit,file=nrcp_file,form='unformatted',access='direct',
     *status='old',recl=rec_length,err=7160)
      read(nrcp_unit,rec=1,err=7170) emin, emax, ne, nb, endian, cdum
      IF (( ichar(endian) .EQ. 0 )) THEN
        endianess = '1234'
      ELSE
        endianess = '4321'
      END IF
      swap = endianess.ne.'1234'
      IF (( swap )) THEN
        tmp_4 = emin
        call egs_swap_4(c_4)
        emin = tmp_4
        tmp_4 = emax
        call egs_swap_4(c_4)
        emax = tmp_4
        itmp_4 = ne
        call egs_swap_4(ic_4)
        ne = itmp_4
        itmp_4 = nb
        call egs_swap_4(ic_4)
        nb = itmp_4
      END IF
      write(i_log,'(//a,a)') 'Reading NRC pair data base from ',nrcp_fil
     *e(:lnblnk1(nrcp_file))
      write(i_log,'(a,a,a)') 'Data generated on a machine with ',endiane
     *ss,' endianess'
      write(i_log,'(a,a)') 'The endianess of this CPU is ','1234'
      IF (( swap )) THEN
        write(i_log,'(a)') '=> will need to do byte swaping'
      END IF
      write(i_log,'(a,2f9.3)') 'Energy range of the data: ',emin,emax
      IF (( nb .NE. 65 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'Inconsistent x-grid size'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      IF (( ne .NE. 84 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'Inconsistent energy grid size'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      nrcp_emin = emin
      nrcp_emax = emax
      nrcp_dle = log((emax-2)/(emin-2))/(ne-1)
      nrcp_dlei = 1/nrcp_dle
      nbb = nb/2
      ddx = sqrt(0.5)/nbb
      DO 7181 ix=0,nbb
        xx = ddx*ix
        nrcp_xdata(ix+1) = xx*xx
7181  CONTINUE
7182  CONTINUE
      do ix=nbb-1,0,-1
        xx = ddx*ix
        nrcp_xdata(nb-ix) = 1 - xx*xx
      end do
      DO 7201 medium=1,NMED
        write(i_log,'(a,i4,a,$)') '  medium ',medium,' .................
     *.... '
        DO 7211 ie=1,84
          DO 7221 ix=1,65
            nrcp_fdata(ix,ie,medium) = 0
7221      CONTINUE
7222      CONTINUE
7211    CONTINUE
7212    CONTINUE
        DO 7231 i_ele=1,NNE(medium)
          Z = ZELEM(medium,i_ele)
          iz = int(Z+0.5)
          tmp = PZ(medium,i_ele)*Z*Z
          irec = (iz-1)*ne + 2
          DO 7241 ie=1,84
            read(nrcp_unit,rec=irec,err=7170) tarray
            DO 7251 ix=1,65
              tmp_4 = tarray(ix)
              IF (( swap )) THEN
                call egs_swap_4(c_4)
              END IF
              nrcp_fdata(ix,ie,medium)=nrcp_fdata(ix,ie,medium)+tmp*tmp_
     *        4
7251        CONTINUE
7252        CONTINUE
            irec = irec + 1
7241      CONTINUE
7242      CONTINUE
7231    CONTINUE
7232    CONTINUE
        DO 7261 ie=1,84
          call prepare_alias_table(nb-1,nrcp_xdata,nrcp_fdata(1,ie,mediu
     *    m), nrcp_wdata(1,ie,medium),nrcp_idata(1,ie,medium))
7261    CONTINUE
7262    CONTINUE
        write(i_log,'(a)') ' done'
7201  CONTINUE
7202  CONTINUE
      write(i_log,*) ' '
      close(nrcp_unit)
      return
7160  CONTINUE
      write(i_log,'(/a)') '***************** Error: '
      write(i_log,*) 'Failed to open NRC pair data file'
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
7170  CONTINUE
      write(i_log,'(/a)') '***************** Error: '
      write(i_log,*) 'I/O error while reading NRC pair data file'
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
      end
      subroutine vmc_electron(ircode)
      implicit none
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      integer*4 ircode
      write(i_log,'(/a)') '***************** Error: '
      write(i_log,'(//a//)') ' ********* VMC Transport option not in thi
     *s distribution ****** '
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
      end
      subroutine egs_init_default_rng
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      call init_ranlux(1,0)
      call ranlux(rng_array)
      rng_seed = 1
      return
      end
      subroutine egs_init_rng(arg1,arg2)
      integer*4 arg1,arg2
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      call init_ranlux(arg1,arg2)
      call ranlux(rng_array)
      rng_seed = 1
      return
      end
      subroutine egs_get_rndm(ran)
      real*8 ran
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      IF (( rng_seed .GT. 24 )) THEN
        call ranlux(rng_array)
        rng_seed = 1
      END IF
      ran = rng_array(rng_seed)
      rng_seed = rng_seed + 1
      return
      end
      subroutine egs_get_rndm_array(n,rarray)
      integer*4 n
      real*8 rarray(*)
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      real*8 rtmp
      integer*4 i
      IF((n .LT. 1))return
      DO 7271 i=1,n
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        rtmp = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        rarray(i) = rtmp
7271  CONTINUE
7272  CONTINUE
      return
      end
      subroutine eii_init
      implicit none
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      COMMON/EDGE/binding_energies(30,100), interaction_prob(6,100), rel
     *axation_prob(39,100), edge_energies(16,100), edge_number(100), edg
     *e_a(16,100), edge_b(16,100), edge_c(16,100), edge_d(16,100), IEDGF
     *L(1),IPHTER(1)
      real*8 binding_energies,  interaction_prob,    relaxation_prob,  e
     *dge_energies,  edge_a,edge_b,edge_c,edge_d
      integer*2 IEDGFL,  IPHTER
      integer*4 edge_number
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      common/eii_data/ eii_xsection_a( 10000),  eii_xsection_b( 10000),
     * eii_cons(1), eii_a(40),  eii_b(40),  eii_L_factor,  eii_z(40),  e
     *ii_sh(40),  eii_nshells(100),  eii_nsh(1),  eii_first(1,50),  eii_
     *no(1,50),  eii_flag
      real*8 eii_xsection_a,eii_xsection_b,eii_a,eii_b,eii_cons,eii_L_fa
     *ctor
      integer*4 eii_z,eii_sh,eii_nshells
      integer*4 eii_first,eii_no
      integer*4 eii_elements,eii_flag,eii_nsh
      COMMON/ELECIN/ esig_e(1),psig_e(1), esige_max, psige_max, range_ep
     *(0:1,500,1), E_array(500,1), etae_ms0(500,1),etae_ms1(500,1),etap_
     *ms0(500,1),etap_ms1(500,1),q1ce_ms0(500,1),q1ce_ms1(500,1),q1cp_ms
     *0(500,1),q1cp_ms1(500,1),q2ce_ms0(500,1),q2ce_ms1(500,1),q2cp_ms0(
     *500,1),q2cp_ms1(500,1),blcce0(500,1),blcce1(500,1), EKE0(1),EKE1(1
     *), XR0(1),TEFF0(1),BLCC(1),XCC(1), ESIG0(500,1),ESIG1(500,1),PSIG0
     *(500,1),PSIG1(500,1),EDEDX0(500,1),EDEDX1(500,1),PDEDX0(500,1),PDE
     *DX1(500,1),EBR10(500,1),EBR11(500,1),PBR10(500,1),PBR11(500,1),PBR
     *20(500,1),PBR21(500,1),TMXS0(500,1),TMXS1(500,1), expeke1(1), IUNR
     *ST(1),EPSTFL(1),IAPRIM(1), sig_ismonotone(0:1,1)
      real*8 esig_e,   psig_e,   esige_max,  psige_max,  range_ep,  E_ar
     *ray,  etae_ms0,etae_ms1,  etap_ms0,etap_ms1,  q1ce_ms0,q1ce_ms1,
     *q1cp_ms0,q1cp_ms1,  q2ce_ms0,q2ce_ms1,  q2cp_ms0,q2cp_ms1,  blcce0
     *,blcce1,   expeke1,  EKE0,EKE1, XR0,  TEFF0,  BLCC,  XCC,  ESIG0,E
     *SIG1,  PSIG0,PSIG1,  EDEDX0,EDEDX1,  PDEDX0,PDEDX1,  EBR10,EBR11,
     * PBR10,PBR11,  PBR20,PBR21,  TMXS0,TMXS1
      integer*4 IUNRST,  EPSTFL,  IAPRIM
      logical sig_ismonotone
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      integer*4 imed,iele,ish,nsh,iZ,j,i,itmp,nskip,nbin,ii,nsh_tot,iii,
     *k
      integer*4 jj,jjj
      integer*4 lnblnk1
      integer*4 tmp_array(100)
      integer*4 want_eii_unit,eii_unit,eii_out,egs_open_file
      integer egs_get_unit
      real*8 e_eii_min,emax,fmax,aux_array(250)
      real*8 sigo,loge,tau,beta2,p2,uwm,Wmax
      real*8 ss_0, ss_1, sh_0, sh_1, aux, av_e, con_med, dedx_old, sigm_
     *old
      real*8 dedx,e,sig,sigm,wbrem,sum_a,sum_z,sum_pz,sum_wa,Ec,Ecc
      real*8 sum_sh,sum_occn,U,sum_sigma,sum_dedx
      real*8 sigma,sigma_old,wbrem_old,sig_j,de
      integer*4 lloge
      logical check_it,is_monotone,getd
      real*8 sigma_max
      character eii_file*128
      character*512 toUpper
      integer*4 occn_numbers(4)
      real*8 cons
      parameter (cons = 0.153536)
      data occn_numbers/2,2,2,4/
      DO 7281 j=1,100
        eii_nshells(j) = 0
7281  CONTINUE
7282  CONTINUE
      DO 7291 j=1,1
        eii_nsh(j) = 0
7291  CONTINUE
7292  CONTINUE
      IF (( eii_flag .EQ. 0 )) THEN
        return
      END IF
      getd = .false.
      DO 7301 j=1,1
        IF (( iedgfl(j) .GT. 0 .AND. iedgfl(j) .LE. 100 )) THEN
          getd = .true.
          GO TO7302
        END IF
7301  CONTINUE
7302  CONTINUE
      IF (( .NOT.getd )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,'(/a,/a,/a,/a)') ' In subroutine eii_init: ', '   Sc
     *attering off bound electrons creates atomic vacancies,', '   poten
     *tially starting an atomic relaxation cascade. ', '   Please turn O
     *N atomic relaxations.'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      e_eii_min = 1e30
      DO 7311 imed=1,nmed
        IF((ae(imed)-rm .LT. e_eii_min))e_eii_min = ae(imed) - rm
        IF((ap(imed) .LT. e_eii_min))e_eii_min = ap(imed)
7311  CONTINUE
7312  CONTINUE
      write(i_log,*) ' '
      write(i_log,*) 'eii_init: minimum threshold energy found: ',e_eii_
     *min
      DO 7321 imed=1,nmed
        DO 7331 iele=1,nne(imed)
          iZ = int(zelem(imed,iele)+0.5)
          IF (( eii_nshells(iZ) .EQ. 0 )) THEN
            nsh = 0
            DO 7341 ish=1,4
              IF((binding_energies(ish,iZ) .GT. e_eii_min))nsh = nsh+1
7341        CONTINUE
7342        CONTINUE
            eii_nshells(iZ) = nsh
          END IF
7331    CONTINUE
7332    CONTINUE
7321  CONTINUE
7322  CONTINUE
      nsh = 0
      DO 7351 iZ=1,100
        nsh = nsh + eii_nshells(iZ)
7351  CONTINUE
7352  CONTINUE
      IF (( nsh .EQ. 0 )) THEN
        write(i_log,*) '*** EII requested but no shells with binding ene
     *rgies '
        write(i_log,*) '    above the specified threshold found'
        write(i_log,*) '    => turning off EII'
        eii_flag = 0
      END IF
      IF (( nsh .GT. 40 )) THEN
        write(i_log,*) '*** Number of shells with binding energies great
     *er than '
        write(i_log,*) '    the specified thresholds is ',nsh
        write(i_log,*) '    This is more than the allocated arrays can h
     *old'
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) '    Increase the macro $MAX_EII_SHELLS and retry
     *'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      write(i_log,*) 'eii_init: number of shells to simulate EII: ',nsh
      nsh_tot = nsh
      tmp_array(1) = 0
      DO 7361 j=2,100
        tmp_array(j) = tmp_array(j-1) + eii_nshells(j-1)
7361  CONTINUE
7362  CONTINUE
      DO 7371 imed=1,nmed
        nsh = 0
        DO 7381 iele=1,nne(imed)
          iZ = int(zelem(imed,iele)+0.5)
          eii_no(imed,iele) = eii_nshells(iZ)
          nsh = nsh + eii_nshells(iZ)
          IF (( eii_nshells(iZ) .GT. 0 )) THEN
            eii_first(imed,iele) = tmp_array(iZ) + 1
          ELSE
            eii_first(imed,iele) = 0
          END IF
7381    CONTINUE
7382    CONTINUE
        eii_nsh(imed) = nsh
7371  CONTINUE
7372  CONTINUE
      DO 7391 i=1,len(eii_file)
        eii_file(i:i) = ' '
7391  CONTINUE
7392  CONTINUE
      eii_file = hen_house(:lnblnk1(hen_house)) // 'data' // '/' // 'eii
     *_'// eii_xfile(:lnblnk1(eii_xfile)) //'.data'
      want_eii_unit = 62
      eii_unit = egs_get_unit(want_eii_unit)
      IF (( eii_unit .LT. 1 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'eii_init: failed to get a free Fortran I/O unit'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      open(eii_unit,file=eii_file(:lnblnk1(eii_file)),status='old',err=7
     *400)
      write(i_log,'(//a,a)') 'Opened EII data file ',eii_file(:lnblnk1(e
     *ii_file))
      write(i_log,'(a,$)') ' eii_init: reading EII data ... '
      read(eii_unit,*,err=7410,end=7410) nskip
      DO 7421 j=1,nskip
        read(eii_unit,*,err=7410,end=7410)
7421  CONTINUE
7422  CONTINUE
      read(eii_unit,*,err=7410,end=7410) emax,nbin
      IF (( nbin .NE. 250 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'Inconsistent EII data file'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      IF ((xsec_out .EQ. 1)) THEN
        eii_out = egs_open_file(93,0,1,'.eiixsec')
      END IF
      ii = 0
      DO 7431 j=1,100
        read(eii_unit,*,err=7410,end=7410) iZ,nsh
        IF ((xsec_out .EQ. 1 .AND. eii_nshells(iZ) .GT. 0)) THEN
          write(eii_out,*) '================================='
          write(eii_out,'(a,i3)') 'EII xsections for element Z = ',iZ
          write(eii_out,*) '================================='
        END IF
        IF (( nsh .LT. eii_nshells(iZ) )) THEN
          write(i_log,*) 'EII data file has data for ',nsh,' shells for
     *element '
          write(i_log,*) iZ,' but according'
          write(i_log,*) 'to binding energies and thresholds ',eii_nshel
     *    ls(iZ)
          write(i_log,*) 'shells are required'
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,*) 'This is a fatal error.'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        DO 7441 ish=1,nsh
          read(eii_unit,*,err=7410,end=7410) fmax
          read(eii_unit,*,err=7410,end=7410) aux_array
          IF ((ish.GT.1 .AND. ish .LT. 5)) THEN
            fmax = fmax*eii_L_factor
          END IF
          IF (( ish .LE. eii_nshells(iZ) )) THEN
            IF ((xsec_out .EQ. 1)) THEN
              IF ((ish .EQ. 1)) THEN
                write(eii_out,'(a,f10.2,a)') 'K-shell sigma_max = ',fmax
     *          ,' b/atom'
              ELSE IF((ish .EQ. 2)) THEN
                write(eii_out,'(a,f9.2,a)') '=> LI-shell sigma_max = ',f
     *          max,' b/atom'
              ELSE IF((ish .EQ. 3)) THEN
                write(eii_out,'(a,f8.2,a)') '=> LII-shell sigma_max = ',
     *          fmax,' b/atom'
              ELSE IF((ish .EQ. 4)) THEN
                write(eii_out,'(a,f8.2,a)') '=> LIII-shell sigma_max = '
     *          ,fmax,' b/atom'
              ELSE
                write(eii_out,*) '=> Wrong number of shells!'
              END IF
              write(eii_out,*) '   E/keV     sigma/(b/atom)'
              write(eii_out,*) '---------------------------'
            END IF
            ii = ii+1
            eii_z(ii) = iZ
            eii_sh(ii) = ish
            eii_a(ii) = nbin
            eii_a(ii) = eii_a(ii)/log(emax/binding_energies(ish,iZ))
            eii_b(ii) = 1 - eii_a(ii)*log(binding_energies(ish,iZ))
            DO 7451 k=1,nbin
              IF (( k .GT. 1 )) THEN
                sigo = fmax*aux_array(k-1)
              ELSE
                sigo = 0
              END IF
              loge = (k - eii_b(ii))/eii_a(ii)
              iii = nbin*(ii-1)+k
              eii_xsection_a(iii) = (fmax*aux_array(k)-sigo)*eii_a(ii)
              eii_xsection_b(iii) = sigo - eii_xsection_a(iii)*loge
              IF ((xsec_out .EQ. 1)) THEN
                write(eii_out,'(f12.2,2X,10f9.2)') Exp((k+1-eii_b(ii))/e
     *          ii_a(ii))*1000.0,fmax*aux_array(k)
              END IF
7451        CONTINUE
7452        CONTINUE
          END IF
7441    CONTINUE
7442    CONTINUE
        IF (( ii .EQ. nsh_tot )) THEN
          GO TO7432
        END IF
7431  CONTINUE
7432  CONTINUE
      close(eii_unit)
      IF ((xsec_out .EQ. 1)) THEN
        close(eii_out)
      END IF
      write(i_log,*) ' OK '
      write(i_log,*) ' '
      DO 7461 imed=1,nmed
        Ec = ae(imed) - rm
        Ecc = min(Ec,ap(imed))
        sum_z=0
        sum_pz=0
        sum_a=0
        sum_wa=0
        DO 7471 iele=1,nne(imed)
          sum_z = sum_z + pz(imed,iele)*zelem(imed,iele)
          sum_pz = sum_pz + pz(imed,iele)
          sum_wa = sum_wa + rhoz(imed,iele)
          sum_a = sum_a + pz(imed,iele)*wa(imed,iele)
7471    CONTINUE
7472    CONTINUE
        con_med = rho(imed)/1.6605655/sum_a
        eii_cons(imed) = con_med
        IF (( eii_nsh(imed) .GT. 0 )) THEN
          is_monotone = .true.
          sigma_max = 0
          DO 7481 j=1,meke(imed)
            loge = (j - eke0(imed))/eke1(imed)
            e = Exp(loge)
            tau = e/rm
            beta2 = tau*(tau+2)/(tau+1)**2
            p2 = 2*rm*tau*(tau+2)
            lloge = j
            medium = imed
            dedx=ededx1(Lloge,MEDIUM)*loge+ededx0(Lloge,MEDIUM)
            IF (( e .GT. ap(medium) .OR. e .GT. 2*Ec )) THEN
              sig=esig1(Lloge,MEDIUM)*loge+esig0(Lloge,MEDIUM)
            ELSE
              sig = 0
            END IF
            IF (( e .GT. 2*Ec )) THEN
              wbrem=ebr11(Lloge,MEDIUM)*loge+ebr10(Lloge,MEDIUM)
              sigm = sig*(1-wbrem)
            ELSE
              sigm = 0
              wbrem = 1
            END IF
            sum_occn=0
            sum_sigma=0
            sum_dedx=0
            DO 7491 iele=1,nne(imed)
              iZ = int(zelem(imed,iele)+0.5)
              sum_sh = 0
              DO 7501 ish=1,eii_no(imed,iele)
                jj = eii_first(imed,iele) + ish - 1
                jjj = eii_sh(jj)
                U = binding_energies(jjj,iZ)
                Wmax = (e+U)/2
                uwm = U/Wmax
                IF (( U .LT. e .AND. U .GT. Ecc )) THEN
                  sum_sh = sum_sh + occn_numbers(jjj)
                  ss_0 = 2*(log(p2/U)-uwm**3*log(p2/Wmax)- (beta2+0.8333
     *            33)*(1-uwm**3))/3/U
                  sh_0 = ((1-uwm)*(1+uwm/(2-uwm))+U*(Wmax-U)/(e+rm)**2 -
     *             (2*tau+1)/(tau+1)**2*uwm/2*log((2-uwm)/uwm))/U
                  ss_1 = log(p2/U)-uwm**2*log(p2/Wmax)- (beta2+1)*(1-uwm
     *            **2)
                  sh_1 = log(Wmax/U/(2-uwm))+2*(Wmax-U)/(2*Wmax-U) +(Wma
     *            x**2-U**2)/(e+rm)**2/2 -(2*tau+1)/(tau+1)**2*log((2*Wm
     *            ax-U)/Wmax)
                  av_E = (ss_1 + sh_1)/(ss_0 + sh_0)
                  i = eii_a(jjj)*loge + eii_b(jjj)
                  i = (jj-1)*250 + i
                  sig_j = eii_xsection_a(i)*loge + eii_xsection_b(i)
                  sig_j = sig_j*pz(imed,iele)*con_med
                  sum_sigma = sum_sigma + sig_j
                  sum_dedx = sum_dedx + sig_j*av_E
                END IF
7501          CONTINUE
7502          CONTINUE
              sum_occn = sum_occn + sum_sh*pz(imed,iele)
7491        CONTINUE
7492        CONTINUE
            sigm = sigm + sum_sigma
            dedx = dedx - sum_dedx
            aux = Ec/e
            IF (( e .GT. 2*Ec )) THEN
              sigo = cons*sum_occn*rho(imed)/(beta2*Ec)*( (1-2*aux)*(1+a
     *        ux/(1-aux)+(tau/(tau+1))**2*aux/2)- (2*tau+1)/(tau+1)**2*a
     *        ux*log((1-aux)/aux))/sum_a
              de = cons*sum_occn*rho(imed)/beta2*( log(0.25/aux/(1-aux))
     *        +(1-2*aux)/(1-aux)+ (tau/(tau+1))**2*(1-4*aux*aux)/8- (2*t
     *        au+1)/(tau+1)**2*log(2*(1-aux)))/sum_a
              sigm = sigm - sigo
              dedx = dedx + de
            END IF
            sigma = sigm + wbrem*sig
            IF((sigma/dedx .GT. sigma_max))sigma_max = sigma/dedx
            IF (( sigma .GT. 0 )) THEN
              wbrem = wbrem*sig/sigma
            ELSE
              wbrem = 1
            END IF
            IF (( j .GT. 1 )) THEN
              ededx1(j-1,imed) = (dedx - dedx_old)*eke1(imed)
              ededx0(j-1,imed) = dedx - ededx1(j-1,imed)*loge
              esig1(j-1,imed) = (sigma - sigma_old)*eke1(imed)
              esig0(j-1,imed) = sigma - esig1(j-1,imed)*loge
              ebr11(j-1,imed) = (wbrem - wbrem_old)*eke1(imed)
              ebr10(j-1,imed) = wbrem - ebr11(j-1,imed)*loge
              IF((sigma/dedx .LT. sigma_old/dedx_old))is_monotone = .fal
     *        se.
            END IF
            dedx_old = dedx
            sigm_old = sigm
            sigma_old = sigma
            wbrem_old = wbrem
7481      CONTINUE
7482      CONTINUE
          ededx1(meke(imed),imed) = ededx1(meke(imed)-1,imed)
          ededx0(meke(imed),imed) = ededx0(meke(imed)-1,imed)
          esig1(meke(imed),imed) = esig1(meke(imed)-1,imed)
          esig0(meke(imed),imed) = esig0(meke(imed)-1,imed)
          ebr11(meke(imed),imed) = ebr11(meke(imed)-1,imed)
          ebr10(meke(imed),imed) = ebr10(meke(imed)-1,imed)
          write(i_log,*) 'eii_init: for medium ',imed,' adjusted sige =
     *', sigma_max,' monotone = ',is_monotone
          sig_ismonotone(0,imed) = is_monotone
          esig_e(imed) = sigma_max
        END IF
7461  CONTINUE
7462  CONTINUE
      return
7410  write(i_log,'(/a)') '***************** Error: '
      write(i_log,*) 'I/O error while reading EII data'
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
7400  write(i_log,'(/a)') '***************** Error: '
      write(i_log,'(//a,a,/a,/a/)') 'Failed to open EII data file ',eii_
     *file(:lnblnk1(eii_file)), 'Make sure file exists in your $HEN_HOUS
     *E/data directory!', '****BEWARE of case sensitive file names!!!'
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
      return
      end
      subroutine eii_sample(ish,iZ,Uj)
      implicit none
      integer*4 ish,iZ
      real*8 Uj
      COMMON/EPCONT/EDEP,EDEP_LOCAL,TSTEP,TUSTEP,USTEP,TVSTEP,VSTEP, RHO
     *F,EOLD,ENEW,EKE,ELKE,GLE,E_RANGE, x_final,y_final,z_final, u_final
     *,v_final,w_final, IDISC,IROLD,IRNEW,IAUSFL(35)
      DOUBLE PRECISION EDEP,  EDEP_LOCAL
      real*8 TSTEP,  TUSTEP,  USTEP,  VSTEP,  TVSTEP,  RHOF,  EOLD,  ENE
     *W,  EKE,  ELKE,  GLE,  E_RANGE, x_final,y_final,z_final,  u_final,
     *v_final,w_final
      integer*4 IDISC,  IROLD,  IRNEW,  IAUSFL
      common/eii_data/ eii_xsection_a( 10000),  eii_xsection_b( 10000),
     * eii_cons(1), eii_a(40),  eii_b(40),  eii_L_factor,  eii_z(40),  e
     *ii_sh(40),  eii_nshells(100),  eii_nsh(1),  eii_first(1,50),  eii_
     *no(1,50),  eii_flag
      real*8 eii_xsection_a,eii_xsection_b,eii_a,eii_b,eii_cons,eii_L_fa
     *ctor
      integer*4 eii_z,eii_sh,eii_nshells
      integer*4 eii_first,eii_no
      integer*4 eii_elements,eii_flag,eii_nsh
      common/egs_vr/ e_max_rr(1),  prob_RR,  nbr_split,  i_play_RR,
     * i_survived_RR,
     *     n_RR_warning,                                        i_do_rr(
     *1)
      real*8          e_max_rr,prob_RR
      integer*4       nbr_split,i_play_RR,i_survived_RR,n_RR_warning
      integer*2     i_do_rr
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      COMMON/UPHIOT/THETA,SINTHE,COSTHE,SINPHI, COSPHI,PI,TWOPI,PI5D2
      real*8 THETA,  SINTHE,  COSTHE,  SINPHI,  COSPHI,  PI,TWOPI,PI5D2
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      common/relax_data/ relax_first(3000),  relax_ntran(3000),  relax_s
     *tate(10000),  relax_prob(10000),  relax_atbin(10000),  relax_ntot
      real*8 relax_prob
      integer*4 relax_first, relax_ntran, relax_state, relax_atbin, rela
     *x_ntot
      real*8 T,tau,tau1,tau12,tau2,p2,beta2,c1,c2,Wmax,xmax,fm_s,fm_h,pr
     *ob_s,prob
      real*8 r1,r2,r3,wx,wxx,aux,frej
      real*8 peie,pese1,pese2,dcosth,h1
      integer*4 iarg
      real*8 eta,cphi,sphi
      integer*4 np_save,ip,j
      real*8 xphi,xphi2,yphi,yphi2,rhophi2
      peie = e(np)
      T = peie - rm
      tau = T/rm
      tau1 = tau+1
      tau12 = tau1*tau1
      tau2 = tau*tau
      p2 = tau2 + 2*tau
      beta2 = p2/tau12
      Wmax = 0.5*(T+Uj)
      xmax = Uj/Wmax
      c1 = (Wmax/peie)**2
      c2 = (2*tau+1)/tau12
      fm_s = log(rmt2*p2/Uj) - beta2 - 0.5
      prob_s = 0.66666667*fm_s*(1+xmax+xmax*xmax)
      fm_h = 2 + c1 - c2
      IF((fm_h .LT. 1))fm_h = 1
      prob = fm_h + prob_s
7511  CONTINUE
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        r1 = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        r2 = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        r3 = rng_array(rng_seed)
        rng_seed = rng_seed + 1
        IF (( r1*prob .LT. fm_h )) THEN
          wx = 1/(r2*xmax+1-r2)
          wxx = wx*xmax
          aux = wxx/(2-wxx)
          frej = (1 + aux*(aux-c2)+c1*wxx*wxx)/fm_h
        ELSE
          wx = 1/(r2*xmax**3+1-r2)**0.333333333
          frej = 1 - log(wx)/fm_s
        END IF
        IF((( r3 .LT. frej )))GO TO7512
      GO TO 7511
7512  CONTINUE
      wx = wx*Uj
      h1 = (peie + prm)/T
      pese1 = peie - wx
      e(np) = pese1
      dcosth = h1*(pese1-prm)/(pese1+prm)
      sinthe = dsqrt(1-dcosth)
      costhe = dsqrt(dcosth)
      call uphi(2,1)
      pese2 = wx - Uj + prm
      edep_local = 0
      IF (( pese2 .GT. ae(medium) )) THEN
        IF (( np+1 .GT. 2 )) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,'(//,3a,/,2(a,i9))') ' In subroutine ','eii_sample
     *', ' stack size exceeded! ',' $MAXSTACK = ',2,' np = ',np+1
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        np = np+1
        e(np) = pese2
        dcosth = h1*(pese2-prm)/(pese2+prm)
        sinthe = -dsqrt(1-dcosth)
        costhe = dsqrt(dcosth)
        iq(np) = -1
        call uphi(3,2)
        edep = 0
      ELSE
        edep = wx - Uj
        edep_local = edep
        IARG=34
        IF ((IAUSFL(IARG+1).NE.0)) THEN
          CALL AUSGAB(IARG)
        END IF
      END IF
      call relax(Uj,ish,iZ)
      IF (( edep .GT. 0 )) THEN
        IARG=4
        IF ((IAUSFL(IARG+1).NE.0)) THEN
          CALL AUSGAB(IARG)
        END IF
      END IF
      return
      end
      subroutine egs_scale_photon_xsection(imed,fac,which)
      implicit none
      integer*4 imed,which
      real*8 fac
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/PHOTIN/ EBINDA(1), GE0(1),GE1(1), GMFP0(2000,1),GMFP1(2000,
     *1),GBR10(2000,1),GBR11(2000,1),GBR20(2000,1),GBR21(2000,1), RCO0(1
     *),RCO1(1), RSCT0(100,1),RSCT1(100,1), COHE0(2000,1),COHE1(2000,1),
     *  PHOTONUC0(2000,1),PHOTONUC1(2000,1), DPMFP, MPGEM(1,1), NGR(1)
      real*8 EBINDA,  GE0,GE1,  GMFP0,GMFP1,  GBR10,GBR11,  GBR20,GBR21,
     *  RCO0,RCO1,  RSCT0,RSCT1,  COHE0,COHE1,   PHOTONUC0,PHOTONUC1,  D
     *PMFP
      integer*4
     *                  MPGEM,
     *                          NGR
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      integer*4 ifirst,ilast,medium,j
      logical has_r
      real*8 gle,gmfp,gbr1,gbr2,cohfac,aux,gmfp_old,gbr1_old,gbr2_old,co
     *hfac_old
      character*8 strings(5)
      data strings/'photon','Rayleigh','Compton','pair','photo'/
      IF (( which .LT. 0 .OR. which .GT. 4 )) THEN
        return
      END IF
      IF (( imed .GT. 0 .AND. imed .LE. nmed )) THEN
        ifirst = imed
        ilast = imed
      ELSE
        ifirst = 1
        ilast = nmed
      END IF
      IF (( which .EQ. 1 )) THEN
        has_r = .false.
        DO 7521 medium=ifirst,ilast
          IF (( iraylm(medium) .EQ. 1 )) THEN
            has_r = .true.
          END IF
7521    CONTINUE
7522    CONTINUE
        IF((.NOT.has_r))return
      END IF
      write(i_log,*) ' '
      DO 7531 medium=ifirst,ilast
        write(i_log,'(a,a,a,i3,a,f9.5)') 'Scaling ',strings(which+1),' x
     *-section data for medium', medium,' with ',fac
        DO 7541 j=1,mge(medium)
          gle = (j - ge0(medium))/ge1(medium)
          gmfp = gmfp0(j,medium) + gmfp1(j,medium)*gle
          gbr1 = gbr10(j,medium) + gbr11(j,medium)*gle
          gbr2 = gbr20(j,medium) + gbr21(j,medium)*gle
          IF (( iraylm(medium) .EQ. 1 )) THEN
            cohfac = cohe0(j,medium) + cohe1(j,medium)*gle
          ELSE
            cohfac = 1
          END IF
          IF (( which .EQ. 0 )) THEN
            gmfp = gmfp/fac
          ELSE IF(( which .EQ. 1 )) THEN
            cohfac = cohfac/(fac*(1-cohfac)+cohfac)
          ELSE
            IF (( which .EQ. 2 )) THEN
              aux = fac*(gbr2-gbr1) + gbr1 + 1 - gbr2
              gbr2 = (gbr1 + fac*(gbr2-gbr1))/aux
              gbr1 = gbr1/aux
            ELSE IF(( which .EQ. 3 )) THEN
              aux = fac*gbr1 + 1 - gbr1
              gbr2 = (fac*gbr1 + gbr2-gbr1)/aux
              gbr1 = fac*gbr1/aux
            ELSE
              aux = gbr2 + fac*(1-gbr2)
              gbr1 = gbr1/aux
              gbr2 = gbr2/aux
            END IF
            gmfp = gmfp/aux
            cohfac = cohfac*aux/(aux*cohfac + 1 - cohfac)
          END IF
          IF (( j .GT. 1 )) THEN
            gmfp1(j-1,medium) = (gmfp - gmfp_old)*ge1(medium)
            gmfp0(j-1,medium) = gmfp - gmfp1(j-1,medium)*gle
            gbr11(j-1,medium) = (gbr1 - gbr1_old)*ge1(medium)
            gbr10(j-1,medium) = gbr1 - gbr11(j-1,medium)*gle
            gbr21(j-1,medium) = (gbr2 - gbr2_old)*ge1(medium)
            gbr20(j-1,medium) = gbr2 - gbr21(j-1,medium)*gle
            cohe1(j-1,medium) = (cohfac - cohfac_old)*ge1(medium)
            cohe0(j-1,medium) = cohfac - cohe1(j-1,medium)*gle
          END IF
          gmfp_old = gmfp
          gbr1_old = gbr1
          gbr2_old = gbr2
          cohfac_old = cohfac
7541    CONTINUE
7542    CONTINUE
        gmfp1(mge(medium),medium) = gmfp1(mge(medium)-1,medium)
        gmfp0(mge(medium),medium) = gmfp0(mge(medium)-1,medium)
        gbr11(mge(medium),medium) = gbr11(mge(medium)-1,medium)
        gbr10(mge(medium),medium) = gbr10(mge(medium)-1,medium)
        gbr21(mge(medium),medium) = gbr21(mge(medium)-1,medium)
        gbr20(mge(medium),medium) = gbr20(mge(medium)-1,medium)
        cohe1(mge(medium),medium) = cohe1(mge(medium)-1,medium)
        cohe0(mge(medium),medium) = cohe0(mge(medium)-1,medium)
7531  CONTINUE
7532  CONTINUE
      return
      end
      subroutine egs_init_user_photon(prefix,comp_prefix,photonuc_prefix
     *,out)
      implicit none
      character*(*) prefix, comp_prefix,  photonuc_prefix
      integer*4 out
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      COMMON/EDGE/binding_energies(30,100), interaction_prob(6,100), rel
     *axation_prob(39,100), edge_energies(16,100), edge_number(100), edg
     *e_a(16,100), edge_b(16,100), edge_c(16,100), edge_d(16,100), IEDGF
     *L(1),IPHTER(1)
      real*8 binding_energies,  interaction_prob,    relaxation_prob,  e
     *dge_energies,  edge_a,edge_b,edge_c,edge_d
      integer*2 IEDGFL,  IPHTER
      integer*4 edge_number
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/PHOTIN/ EBINDA(1), GE0(1),GE1(1), GMFP0(2000,1),GMFP1(2000,
     *1),GBR10(2000,1),GBR11(2000,1),GBR20(2000,1),GBR21(2000,1), RCO0(1
     *),RCO1(1), RSCT0(100,1),RSCT1(100,1), COHE0(2000,1),COHE1(2000,1),
     *  PHOTONUC0(2000,1),PHOTONUC1(2000,1), DPMFP, MPGEM(1,1), NGR(1)
      real*8 EBINDA,  GE0,GE1,  GMFP0,GMFP1,  GBR10,GBR11,  GBR20,GBR21,
     *  RCO0,RCO1,  RSCT0,RSCT1,  COHE0,COHE1,   PHOTONUC0,PHOTONUC1,  D
     *PMFP
      integer*4
     *                  MPGEM,
     *                          NGR
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      common/compton_data/ iz_array(1538),  be_array(1538),  Jo_array(15
     *38),  erfJo_array(1538),   ne_array(1538),  shn_array(1538),
     *shell_array(200,1), eno_array(200,1), eno_atbin_array(200,1), n_sh
     *ell(1), radc_flag,  ibcmp(1)
      integer*4 iz_array,ne_array,shn_array,eno_atbin_array, shell_array
     *,n_shell,radc_flag
      real*8 be_array,Jo_array,erfJo_array,eno_array
      integer*2 ibcmp
      common/x_options/eadl_relax,  mcdf_pe_xsections
      logical eadl_relax, mcdf_pe_xsections
      integer*4 lnblnk1,egs_get_unit,medium, photo_unit,pair_unit,raylei
     *gh_unit,triplet_unit, ounit,egs_open_file,compton_unit,  photonuc_
     *unit
      integer*4 nge,sorted(50),i,j,k,iz,iz_old,ndat
      real*8 z_sorted(50),pz_sorted(50)
      real*8 sig_photo(2000),sig_pair(2000),sig_triplet(2000), sig_rayle
     *igh(2000),sig_compton(2000)
      real*8 sigma,cohe,gmfp,gbr1,gbr2,sig_KN,gle,e,sig_p
      real*8 cohe_old,gmfp_old,gbr1_old,gbr2_old,  sig_photonuc(2000), p
     *hotonuc, photonuc_old
      real*8 etmp(2000),ftmp(2000)
      real*8 sumZ,sumA,con1,con2,egs_KN_sigma0
      real*8 bc_emin,bc_emax,bc_dle,bc_data(183),bc_tmp(183),bcf,aj
      integer*4 bc_ne
      logical input_compton_data,  input_photonuc_data
      character data_dir*128,photo_file*140,pair_file*140,rayleigh_file*
     *144, triplet_file*142,tmp_string*144,compton_file*144,  photonuc_f
     *ile*144
      write(i_log,'(/a$)') '(Re)-initializing photon cross sections'
      write(i_log,'(a,a/)') ' with files from the series: ', prefix(:lnb
     *lnk1(prefix))
      write(i_log,'(a,a)') ' Compton cross sections: ',comp_prefix(:lnbl
     *nk1(comp_prefix))
      IF ((iphotonuc .EQ. 1)) THEN
        write(i_log,'(a,a)') ' Photonuclear cross sections: ', photonuc_
     *  prefix(:lnblnk1(photonuc_prefix))
        input_photonuc_data = .false.
        IF ((lnblnk1(photonuc_prefix) .GT. 0 .AND. photonuc_prefix(1:7)
     *  .NE. 'default')) THEN
          input_photonuc_data = .true.
        END IF
      END IF
      input_compton_data = .false.
      IF (( ibcmp(1) .GT. 1 .AND. lnblnk1(comp_prefix) .GT. 0 )) THEN
        IF((comp_prefix(1:7) .NE. 'default'))input_compton_data = .true.
      END IF
      data_dir = hen_house(:lnblnk1(hen_house)) // 'data' // '/'
      photo_file = data_dir(:lnblnk1(data_dir)) // prefix(:lnblnk1(prefi
     *x)) // '_photo.data'
      pair_file = data_dir(:lnblnk1(data_dir)) // prefix(:lnblnk1(prefix
     *)) // '_pair.data'
      triplet_file = data_dir(:lnblnk1(data_dir)) // prefix(:lnblnk1(pre
     *fix)) // '_triplet.data'
      rayleigh_file = data_dir(:lnblnk1(data_dir)) // prefix(:lnblnk1(pr
     *efix)) // '_rayleigh.data'
      IF (( input_compton_data )) THEN
        compton_file = data_dir(:lnblnk1(data_dir)) // comp_prefix(:lnbl
     *  nk1(comp_prefix)) // '_compton.data'
      ELSE
        compton_file = data_dir(:lnblnk1(data_dir)) // 'compton_sigma.da
     *ta'
      END IF
      write(i_log,'(a,a)') ' Using Compton cross sections from ', compto
     *n_file(:lnblnk1(compton_file))
      IF ((iphotonuc .EQ. 1)) THEN
        IF (( input_photonuc_data )) THEN
          photonuc_file = data_dir(:lnblnk1(data_dir)) // photonuc_prefi
     *    x(:lnblnk1(photonuc_prefix)) // '_photonuc.data'
        ELSE
          photonuc_file = data_dir(:lnblnk1(data_dir)) // 'iaea_photonuc
     *.data'
        END IF
        write(i_log,'(a,a)') ' Using photonuclear cross sections from ',
     *   photonuc_file(:lnblnk1(photonuc_file))
      END IF
      photo_unit = 83
      photo_unit = egs_get_unit(photo_unit)
      IF (( photo_unit .LT. 1 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'egs_init_user_photon: failed to get a free Fortr
     *an I/O unit'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      tmp_string = photo_file
      open(photo_unit,file=photo_file,status='old',err=7550)
      pair_unit = 84
      pair_unit = egs_get_unit(pair_unit)
      IF (( pair_unit .LT. 1 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'egs_init_user_photon: failed to get a free Fortr
     *an I/O unit'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      tmp_string = pair_file
      open(pair_unit,file=pair_file,status='old',err=7550)
      triplet_unit = 85
      triplet_unit = egs_get_unit(triplet_unit)
      IF (( triplet_unit .LT. 1 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'egs_init_user_photon: failed to get a free Fortr
     *an I/O unit'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      tmp_string = triplet_file
      open(triplet_unit,file=triplet_file,status='old',err=7550)
      rayleigh_unit = 86
      rayleigh_unit = egs_get_unit(rayleigh_unit)
      IF (( rayleigh_unit .LT. 1 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'egs_init_user_photon: failed to get a free Fortr
     *an I/O unit'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      tmp_string = rayleigh_file
      open(rayleigh_unit,file=rayleigh_file,status='old',err=7550)
      IF (( ibcmp(1) .GT. 1 )) THEN
        compton_unit = 88
        compton_unit = egs_get_unit(compton_unit)
        IF (( compton_unit .LT. 1 )) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,*) 'egs_init_user_photon: failed to get a free For
     *tran I/O unit'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        tmp_string = compton_file
        open(compton_unit,file=compton_file,status='old',err=7550)
      END IF
      IF (( iphotonuc .EQ. 1 )) THEN
        photonuc_unit = 89
        photonuc_unit = egs_get_unit(photonuc_unit)
        IF (( photonuc_unit .LT. 1 )) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,*) 'egs_init_user_photon: failed to get a free For
     *tran I/O unit'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        tmp_string = photonuc_file
        open(photonuc_unit,file=photonuc_file,status='old',err=7550)
      END IF
      IF (( out .EQ. 1 )) THEN
        ounit = egs_open_file(87,0,1,'.xsections')
        write(ounit,'(/a,a,a)') 'Photon cross sections initialized from
     *', prefix(:lnblnk1(prefix)),' data files'
        write(ounit,'(a,/)') '==========================================
     *=================================='
        write(ounit,'(a,/)') 'Grid energies and cross sections are outpu
     *t'
        IF ((iphotonuc .EQ. 1)) THEN
          write(ounit,'(5x,a,t19,a,t34,a,t49,a,t64,a,t79,a)') 'Energy','
     * GMFP(cm) ',' Pair ','Compton',' GMFP(cm) ', ' GMFP(cm) '
          write(ounit,'(5x,a,t19,a,t34,a,t49,a,t64,a,t79,a/)') '(MeV)','
     *no Rayleigh','(fraction)','(fraction)','with Rayleigh', 'w/ Ray +
     *photnuc'
        ELSE
          write(ounit,'(5x,a,t19,a,t34,a,t49,a,t64,a)') 'Energy',' GMFP(
     *cm) ',' Pair ','Compton',' GMFP(cm) '
          write(ounit,'(5x,a,t19,a,t34,a,t49,a,t64,a/)') '(MeV)','no Ray
     *leigh','(fraction)','(fraction)','with Rayleigh'
        END IF
      END IF
      DO 7561 iz=1,100
        read(photo_unit,*) ndat
        read(photo_unit,*) (etmp(k),ftmp(k),k=1,ndat)
        k = 0
        DO 7571 j=ndat,2,-1
          IF (( etmp(j)-etmp(j-1) .LT. 1e-5 )) THEN
            k = k+1
            IF (( k .LE. 30 )) THEN
              binding_energies(k,iz) = exp(etmp(j))
            ELSE
              write(i_log,'(/a)') '***************** Error: '
              write(i_log,'(i3,a,i3,//a)') k,' binding energies read exc
     *eeding array size of', 30,'Increase $MXSHXSEC in egsnrc.macros!'
              write(i_log,'(/a)') '***************** Quiting now.'
              call exit(1)
            END IF
            IF((.NOT.eadl_relax .AND. k .GE. 4))GO TO7572
          END IF
7571    CONTINUE
7572    CONTINUE
7561  CONTINUE
7562  CONTINUE
      IF ((mcdf_pe_xsections)) THEN
        call egs_read_shellwise_pe()
      END IF
      DO 7581 medium=1,nmed
        mge(medium) = 2000
        nge = 2000
        ge1(medium) = nge-1
        ge1(medium) = ge1(medium)/log(up(medium)/ap(medium))
        ge0(medium) = 1 - ge1(medium)*log(ap(medium))
        write(i_log,'(a,i3,a,$)') ' Working on medium ',medium,' ... '
        IF (( out .EQ. 1 )) THEN
          write(ounit,'(/,2x,a,i3,a,24a1/)') 'Medium ',medium,': ', (med
     *    ia(k,medium),k=1,24)
        END IF
        sumZ=0
        sumA=0
        DO 7591 i=1,nne(medium)
          z_sorted(i) = zelem(medium,i)
          sumZ = sumZ + pz(medium,i)*zelem(medium,i)
          sumA = sumA + pz(medium,i)*wa(medium,i)
7591    CONTINUE
7592    CONTINUE
        con1 = sumZ*rho(medium)/(sumA*1.6605655)
        con2 = rho(medium)/(sumA*1.6605655)
        call egs_heap_sort(nne(medium),z_sorted,sorted)
        DO 7601 i=1,nne(medium)
          pz_sorted(i) = pz(medium,sorted(i))
7601    CONTINUE
7602    CONTINUE
        IF ((mcdf_pe_xsections)) THEN
          call egsi_get_shell_data(medium,nge,nne(medium),z_sorted,pz_so
     *    rted, ge1(medium),ge0(medium),sig_photo)
        ELSE
          call egsi_get_data(0,photo_unit,nge,nne(medium),z_sorted,pz_so
     *    rted, ge1(medium),ge0(medium),sig_photo)
        END IF
        call egsi_get_data(0,rayleigh_unit,nge,nne(medium),z_sorted,pz_s
     *  orted, ge1(medium),ge0(medium),sig_rayleigh)
        call egsi_get_data(1,pair_unit,nge,nne(medium),z_sorted,pz_sorte
     *  d, ge1(medium),ge0(medium),sig_pair)
        call egsi_get_data(2,triplet_unit,nge,nne(medium),z_sorted,pz_so
     *  rted, ge1(medium),ge0(medium),sig_triplet)
        IF (( iphotonuc .EQ. 1 )) THEN
          call egsi_get_data(3,photonuc_unit,nge,nne(medium),z_sorted,pz
     *    _sorted, ge1(medium),ge0(medium),sig_photonuc)
        END IF
        IF (( ibcmp(1) .GT. 1 )) THEN
          IF (( input_compton_data )) THEN
            call egsi_get_data(0,compton_unit,nge,nne(medium), z_sorted,
     *      pz_sorted,ge1(medium),ge0(medium), sig_compton)
          ELSE
            rewind(compton_unit)
            read(compton_unit,*) bc_emin,bc_emax,bc_ne
            IF (( bc_ne .GT. 183 )) THEN
              write(i_log,'(/a)') '***************** Error: '
              write(i_log,*) 'Number of input Compton data exceeds array
     * size'
              write(i_log,'(/a)') '***************** Quiting now.'
              call exit(1)
            END IF
            bc_dle = log(bc_emax/bc_emin)/(bc_ne-1)
            DO 7611 j=1,bc_ne
              bc_data(j) = 0
7611        CONTINUE
7612        CONTINUE
            iz_old = 1
            DO 7621 i=1,nne(medium)
              iz = int(z_sorted(i)+0.5)
              DO 7631 j=iz_old,iz
                read(compton_unit,*) (bc_tmp(k),k=1,bc_ne)
7631          CONTINUE
7632          CONTINUE
              DO 7641 j=1,bc_ne
                bc_data(j)=bc_data(j)+pz_sorted(i)*z_sorted(i)*bc_tmp(j)
7641          CONTINUE
7642          CONTINUE
              iz_old = iz+1
7621        CONTINUE
7622        CONTINUE
            DO 7651 j=1,bc_ne
              bc_data(j)=log(bc_data(j)/sumZ)
7651        CONTINUE
7652        CONTINUE
          END IF
        END IF
        call egs_init_rayleigh(medium,sig_rayleigh)
        DO 7661 i=1,nge
          gle = (i - ge0(medium))/ge1(medium)
          e = exp(gle)
          sig_KN = sumZ*egs_KN_sigma0(e)
          IF (( ibcmp(1) .GT. 1 )) THEN
            IF (( input_compton_data )) THEN
              sig_KN = sig_compton(i)
            ELSE
              IF (( e .LE. bc_emin )) THEN
                bcf = exp(bc_data(1))
              ELSE IF(( e .LT. bc_emax )) THEN
                aj = 1 + log(e/bc_emin)/bc_dle
                j = int(aj)
                aj = aj - j
                bcf = exp(bc_data(j)*(1-aj) + bc_data(j+1)*aj)
              ELSE
                bcf = 1
              END IF
              sig_KN = sig_KN*bcf
            END IF
          END IF
          sig_p = sig_pair(i) + sig_triplet(i)
          sigma = sig_KN + sig_p + sig_photo(i)
          gmfp = 1/(sigma*con2)
          gbr1 = sig_p/sigma
          gbr2 = gbr1 + sig_KN/sigma
          cohe = sigma/(sig_rayleigh(i) + sigma)
          photonuc = sigma/(sig_photonuc(i) + sigma)
          IF (( out .EQ. 1 )) THEN
            IF ((iphotonucm(medium) .EQ. 1)) THEN
              write(ounit,'(6(1pe15.6))') e,gmfp,gbr1,gbr2-gbr1, gmfp*co
     *        he,gmfp*cohe*photonuc
            ELSE
              write(ounit,'(5(1pe15.6))') e,gmfp,gbr1,gbr2-gbr1,gmfp*coh
     *        e
            END IF
          END IF
          IF (( i .GT. 1 )) THEN
            gmfp1(i-1,medium) = (gmfp - gmfp_old)*ge1(medium)
            gmfp0(i-1,medium) = gmfp - gmfp1(i-1,medium)*gle
            gbr11(i-1,medium) = (gbr1 - gbr1_old)*ge1(medium)
            gbr10(i-1,medium) = gbr1 - gbr11(i-1,medium)*gle
            gbr21(i-1,medium) = (gbr2 - gbr2_old)*ge1(medium)
            gbr20(i-1,medium) = gbr2 - gbr21(i-1,medium)*gle
            cohe1(i-1,medium) = (cohe - cohe_old)*ge1(medium)
            cohe0(i-1,medium) = cohe - cohe1(i-1,medium)*gle
            photonuc1(i-1,medium) = (photonuc - photonuc_old)*ge1(medium
     *      )
            photonuc0(i-1,medium) = photonuc - photonuc1(i-1,medium)*gle
          END IF
          gmfp_old = gmfp
          gbr1_old = gbr1
          gbr2_old = gbr2
          cohe_old = cohe
          photonuc_old = photonuc
7661    CONTINUE
7662    CONTINUE
        gmfp1(nge,medium) = gmfp1(nge-1,medium)
        gmfp0(nge,medium) = gmfp - gmfp1(nge,medium)*gle
        gbr11(nge,medium) = gbr11(nge-1,medium)
        gbr10(nge,medium) = gbr1 - gbr11(nge,medium)*gle
        gbr21(nge,medium) = gbr21(nge-1,medium)
        gbr20(nge,medium) = gbr2 - gbr21(nge,medium)*gle
        cohe1(nge,medium) = cohe1(nge-1,medium)
        cohe0(nge,medium) = cohe - cohe1(nge,medium)*gle
        photonuc1(nge,medium) = photonuc1(nge-1,medium)
        photonuc0(nge,medium) = photonuc - photonuc1(nge,medium)*gle
        write(i_log,'(a)') 'OK'
7581  CONTINUE
7582  CONTINUE
      close(photo_unit)
      close(pair_unit)
      close(triplet_unit)
      close(rayleigh_unit)
      IF (( iphotonuc .EQ. 1 )) THEN
        close(photonuc_unit)
      END IF
      IF (( ibcmp(1) .GT. 1 )) THEN
        close(compton_unit)
      END IF
      IF (( out .EQ. 1 )) THEN
        close(ounit)
      END IF
      return
7550  CONTINUE
      write(i_log,'(/a)') '***************** Error: '
      write(i_log,'(//a,a)') 'Failed to open data file ',tmp_string(:lnb
     *lnk1(tmp_string))
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
      return
      end
      subroutine egs_init_rayleigh(medium,sig_rayleigh)
      implicit none
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/MISC/  DUNIT,KMPI,KMPO,RHOR(1),MED(1),IRAYLR(1),IPHOTONUCR(
     *1)
      real*8 DUNIT,  RHOR
      integer*4 KMPI,  KMPO
      integer*2 MED,  IRAYLR,  IPHOTONUCR
      COMMON/PHOTIN/ EBINDA(1), GE0(1),GE1(1), GMFP0(2000,1),GMFP1(2000,
     *1),GBR10(2000,1),GBR11(2000,1),GBR20(2000,1),GBR21(2000,1), RCO0(1
     *),RCO1(1), RSCT0(100,1),RSCT1(100,1), COHE0(2000,1),COHE1(2000,1),
     *  PHOTONUC0(2000,1),PHOTONUC1(2000,1), DPMFP, MPGEM(1,1), NGR(1)
      real*8 EBINDA,  GE0,GE1,  GMFP0,GMFP1,  GBR10,GBR11,  GBR20,GBR21,
     *  RCO0,RCO1,  RSCT0,RSCT1,  COHE0,COHE1,   PHOTONUC0,PHOTONUC1,  D
     *PMFP
      integer*4
     *                  MPGEM,
     *                          NGR
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      COMMON/rayleigh_inputs/iray_ff_media(1),iray_ff_file(1)
      character*24 iray_ff_media
      character*128 iray_ff_file
      COMMON/rayleigh_sampling/xgrid(100,1), fcum(100,1), b_array(100,1)
     *, c_array(100,1), i_array(100,1), pmax0(2000,1),pmax1(2000,1)
      real*8 xgrid, fcum, b_array, c_array,pmax0, pmax1
      integer*4 i_array
      real*8 xval(100),aff(100,100),ff(100,1)
      real*8 xsc, fsc
      real*8 sig_rayleigh(2000), pe_array(2000,1)
      real*8 e,egs_rayleigh_sigma,gmfp,gle,conv,dle,dlei,sumA
      real*8 totRayleigh2,pzmin
      real*8 emin, emax
      integer*4 i,j,k,ff_unit, egs_get_unit, ne
      integer*4 lnblnk1, EOF, nff, medium, ncustom
      character dummy*24, afac_file*128, ff_file*128
      IF ((iraylm(medium).EQ.0)) THEN
        return
      END IF
      ncustom=0
      write(dummy,'(24a1)')(media(j,medium),j=1,24)
      ff_file=' '
      DO 7671 i=1,1
        IF ((lnblnk1(iray_ff_file(i)).NE.0)) THEN
          ncustom = ncustom + 1
        END IF
7671  CONTINUE
7672  CONTINUE
      DO 7681 i=1,ncustom
        IF ((dummy(:lnblnk1(dummy)) .EQ. iray_ff_media(i))) THEN
          ff_file = iray_ff_file(i)
        END IF
7681  CONTINUE
7682  CONTINUE
      ff_unit = egs_get_unit(0)
      IF (( ff_unit .LT. 1 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'egs_init_rayleigh: failed to get a free Fortran
     *I/O unit'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      IF (( lnblnk1(ff_file) .GT. 0)) THEN
        open(ff_unit,file=ff_file(:lnblnk1(ff_file)), status='old',err=7
     *  690)
        GOTO 7700
7690    write(i_log,'(/a)') '***************** Error: '
        write(i_log,'(2a)') 'egs_init_rayleigh: failed to open custom ff
     * file ', ff_file(:lnblnk1(ff_file))
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
7700    write(i_log,'(/2a)') 'Opened custom ff file ',ff_file(:lnblnk1(f
     *  f_file))
        j = 0
7711    CONTINUE
          j = j + 1
          read(ff_unit,*,IOSTAT=EOF) xsc, fsc
          IF((EOF .LT. 0))GO TO7712
          IF ((j .LE. 100)) THEN
            xgrid(j,medium)=xsc
            ff(j,medium)=fsc
          END IF
        GO TO 7711
7712    CONTINUE
        nff = j-1
        IF ((nff .GT. 100)) THEN
          write(i_log,'(/a)') '***************** Error: '
          write(i_log,'(a,/,a,i5,a,i5,/,a)') 'subroutine egs_init_raylei
     *gh: form factors size too small!!', '$XRAYFF =  ', 100,', and need
     * to be ',nff, ' and try again!!!'
          write(i_log,'(/a)') '***************** Quiting now.'
          call exit(1)
        END IF
        IF((xgrid(1,medium) .LT. 1e-6))xgrid(1,medium) = 1e-4
        write(*,*) '\n  -> ', nff, ' values of mol. ff read!'
        sumA = 0.0
        DO 7721 j=1,nne(medium)
          sumA=sumA+PZ(medium,j)*WA(medium,j)
7721    CONTINUE
7722    CONTINUE
        DO 7731 j=1,MGE(medium)
          gle=(j-GE0(medium))/GE1(medium)
          e=exp(gle)
          sig_rayleigh(j)=egs_rayleigh_sigma(medium,e,nff, xgrid(1,mediu
     *    m),ff(1,medium))*sumA
7731    CONTINUE
7732    CONTINUE
      ELSE
        DO 7741 i=1,len(afac_file)
          afac_file(i:i) = ' '
7741    CONTINUE
7742    CONTINUE
        afac_file = hen_house(:lnblnk1(hen_house))//'pegs4'//'/'//'pgs4f
     *orm.dat'
        open(ff_unit,file=afac_file(:lnblnk1(afac_file)), status='old',e
     *  rr=7750)
        GOTO 7760
7750    write(i_log,'(/a)') '***************** Error: '
        write(i_log,'(2a)') 'egs_init_rayleigh: failed to open atomic ff
     * file', afac_file(:lnblnk1(afac_file))
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
7760    read(ff_unit,*) xval, aff
        DO 7771 i=1,100
          ff(i,medium) = 0.0
          xgrid(i,medium)=xval(i)
          DO 7781 j=1,nne(medium)
            ff(i,medium)=ff(i,medium)+PZ(medium,j)*aff(i,int(zelem(mediu
     *      m,j)))**2
7781      CONTINUE
7782      CONTINUE
          ff(i,medium) = sqrt(ff(i,medium))
7771    CONTINUE
7772    CONTINUE
        nff = 100
        IF((xgrid(1,medium) .LT. 1e-6))xgrid(1,medium) = 1e-4
        write(i_log,'(/a,i4,a)') '  -> ', nff, ' atomic ff values comput
     *ed!'
      END IF
      close(ff_unit)
      emin = exp((1 - ge0(medium))/ge1(medium))
      emax = exp((mge(medium) - ge0(medium))/ge1(medium))
      call prepare_rayleigh_data(nff,xgrid(1,medium),ff(1,medium), mge(m
     *edium),emin,emax, pe_array(1,medium),100, fcum(1,medium),i_array(1
     *,medium), b_array(1,medium),c_array(1,medium))
      ne=MGE(medium)
      dle=log(up(medium)/ap(medium))/(ne-1)
      dlei=1/dle
      DO 7791 i=1,ne-1
        gle = (i - ge0(medium))/ge1(medium)
        pmax1(i,medium)=(pe_array(i+1,medium)-pe_array(i,medium))*ge1(me
     *  dium)
        pmax0(i,medium)=pe_array(i,medium)-pmax1(i,medium)*gle
7791  CONTINUE
7792  CONTINUE
      pmax0(ne,medium)=pmax0(ne-1,medium)
      pmax1(ne,medium)=pmax1(ne-1,medium)
      return
      end
      subroutine egs_init_rayleigh_sampling(medium)
      implicit none
      COMMON/THRESH/RMT2,RMSQ, AP(1),AE(1),UP(1),UE(1),TE(1),THMOLL(1)
      real*8 RMT2,  RMSQ,  AP,  AE,  UP,  UE,  TE,  THMOLL
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      COMMON/MISC/  DUNIT,KMPI,KMPO,RHOR(1),MED(1),IRAYLR(1),IPHOTONUCR(
     *1)
      real*8 DUNIT,  RHOR
      integer*4 KMPI,  KMPO
      integer*2 MED,  IRAYLR,  IPHOTONUCR
      COMMON/PHOTIN/ EBINDA(1), GE0(1),GE1(1), GMFP0(2000,1),GMFP1(2000,
     *1),GBR10(2000,1),GBR11(2000,1),GBR20(2000,1),GBR21(2000,1), RCO0(1
     *),RCO1(1), RSCT0(100,1),RSCT1(100,1), COHE0(2000,1),COHE1(2000,1),
     *  PHOTONUC0(2000,1),PHOTONUC1(2000,1), DPMFP, MPGEM(1,1), NGR(1)
      real*8 EBINDA,  GE0,GE1,  GMFP0,GMFP1,  GBR10,GBR11,  GBR20,GBR21,
     *  RCO0,RCO1,  RSCT0,RSCT1,  COHE0,COHE1,   PHOTONUC0,PHOTONUC1,  D
     *PMFP
      integer*4
     *                  MPGEM,
     *                          NGR
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      COMMON/rayleigh_inputs/iray_ff_media(1),iray_ff_file(1)
      character*24 iray_ff_media
      character*128 iray_ff_file
      COMMON/rayleigh_sampling/xgrid(100,1), fcum(100,1), b_array(100,1)
     *, c_array(100,1), i_array(100,1), pmax0(2000,1),pmax1(2000,1)
      real*8 xgrid, fcum, b_array, c_array,pmax0, pmax1
      integer*4 i_array
      real*8 xval(100),aff(100,100),ff(100,1)
      real*8 xsc, fsc
      real*8 sig_rayleigh(2000), pe_array(2000,1)
      real*8 e,egs_rayleigh_sigma,gmfp,gle,conv,dle,dlei,sumA
      real*8 totRayleigh2,pzmin
      real*8 emin, emax
      integer*4 i,j,k,ff_unit, egs_get_unit, ne
      integer*4 lnblnk1, EOF, nff, medium, ncustom
      character dummy*24, afac_file*128, ff_file*128
      IF ((iraylm(medium).EQ.0)) THEN
        return
      END IF
      ff_unit = egs_get_unit(0)
      IF (( ff_unit .LT. 1 )) THEN
        write(i_log,'(/a)') '***************** Error: '
        write(i_log,*) 'egs_init_rayleigh: failed to get a free Fortran
     *I/O unit'
        write(i_log,'(/a)') '***************** Quiting now.'
        call exit(1)
      END IF
      DO 7801 i=1,len(afac_file)
        afac_file(i:i) = ' '
7801  CONTINUE
7802  CONTINUE
      afac_file = hen_house(:lnblnk1(hen_house))//'pegs4'//'/'//'pgs4for
     *m.dat'
      open(ff_unit,file=afac_file(:lnblnk1(afac_file)),status='old',err=
     *7750)
      GOTO 7760
7750  write(i_log,'(/a)') '***************** Error: '
      write(i_log,'(2a)') 'egs_init_rayleigh_sampling: failed to open at
     *omic ff file ', afac_file(:lnblnk1(afac_file))
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
7760  read(ff_unit,*) xval, aff
      DO 7811 i=1,100
        ff(i,medium) = 0.0
        xgrid(i,medium)=xval(i)
        DO 7821 j=1,nne(medium)
          ff(i,medium)=ff(i,medium)+PZ(medium,j)*aff(i,int(zelem(medium,
     *    j)))**2
7821    CONTINUE
7822    CONTINUE
        ff(i,medium) = sqrt(ff(i,medium))
7811  CONTINUE
7812  CONTINUE
      nff = 100
      IF((xgrid(1,medium) .LT. 1e-6))xgrid(1,medium) = 1e-4
      write(i_log,'(/a,i4,a)') '  -> ', nff, ' atomic ff values computed
     *!'
      close(ff_unit)
      emin = exp((1 - ge0(medium))/ge1(medium))
      emax = exp((mge(medium) - ge0(medium))/ge1(medium))
      call prepare_rayleigh_data(nff,xgrid(1,medium),ff(1,medium), mge(m
     *edium),emin,emax, pe_array(1,medium),100, fcum(1,medium),i_array(1
     *,medium), b_array(1,medium),c_array(1,medium))
      ne=MGE(medium)
      DO 7831 i=1,ne-1
        gle = (i - ge0(medium))/ge1(medium)
        pmax1(i,medium)=(pe_array(i+1,medium)-pe_array(i,medium))*ge1(me
     *  dium)
        pmax0(i,medium)=pe_array(i,medium)-pmax1(i,medium)*gle
7831  CONTINUE
7832  CONTINUE
      pmax0(ne,medium)=pmax0(ne-1,medium)
      pmax1(ne,medium)=pmax1(ne-1,medium)
      return
      end
      real*8 function egs_rayleigh_sigma(imed,E,ndat,x,f)
      implicit none
      integer*4 i, j, k,imed, ndat
      real*8 hc2,conv,b,hc
      parameter (hc = 0.0123984768438,hc2=0.0001537222280)
      real*8 x(100), f(100), zero, E, xmax
      real*8 x1,x2,pow_x1,pow_x2,raysig,C,C2,f1,f2
      C=2.*hc2/(E*E)
      C2=C*C
      xmax=E/hc
      egs_rayleigh_sigma = 0.0
      DO 7841 i=1,ndat-1
        IF((x(i) .EQ. 0.0))x(i) = zero()
        IF((x(i+1) .EQ. 0.0))x(i+1) = zero()
        IF((f(i) .EQ. 0.0))f(i) = zero()
        IF((f(i+1) .EQ. 0.0))f(i+1) = zero()
        b = log(f(i+1)/f(i))/log(x(i+1)/x(i))
        x1=x(i)
        x2=x(i+1)
        IF ((x2 .GT. xmax)) THEN
          x2=xmax
        END IF
        pow_x1=x1**(2*b)
        pow_x2=x2**(2*b)
        raysig = pow_x2*(x2**2/(b+1)-(C*x2**4)/(b+2)+(C2*x2**6)/(2*b+6))
        raysig = raysig - pow_x1*(x1**2/(b+1)-(C*x1**4)/(b+2)+(C2*x1**6)
     *  /(2*b+6))
        raysig = raysig*f(i)*f(i)/pow_x1
        egs_rayleigh_sigma = egs_rayleigh_sigma + raysig
        IF ((x(i+1).GT.xmax)) THEN
          GO TO7842
        END IF
7841  CONTINUE
7842  CONTINUE
      egs_rayleigh_sigma = 0.49893439187842413747*C*egs_rayleigh_sigma
      return
      end
      subroutine egs_rayleigh_sampling(medium,e,gle,lgle,costhe,sinthe)
      implicit none
      real*8 e
      real*8 gle,costhe,sinthe,pmax,xv,xmax,csqthe
      real*8 rnnray1,rnnray0,hc_i,twice_hc2,dwi
      parameter(hc_i=80.65506856998,twice_hc2=0.000307444456)
      integer*4 lgle,ib,ibin,medium, trials
      common/randomm/ rng_array(24), seeds(25), rng_seed
      real*8 rng_array
      integer*4 rng_seed,  seeds
      COMMON/rayleigh_sampling/xgrid(100,1), fcum(100,1), b_array(100,1)
     *, c_array(100,1), i_array(100,1), pmax0(2000,1),pmax1(2000,1)
      real*8 xgrid, fcum, b_array, c_array,pmax0, pmax1
      integer*4 i_array
      dwi = 100-1
      pmax=pmax1(Lgle,MEDIUM)*gle+pmax0(Lgle,MEDIUM)
      xmax = hc_i*e
7851  CONTINUE
        IF (( rng_seed .GT. 24 )) THEN
          call ranlux(rng_array)
          rng_seed = 1
        END IF
        rnnray1 = rng_array(rng_seed)
        rng_seed = rng_seed + 1
7861    CONTINUE
          IF (( rng_seed .GT. 24 )) THEN
            call ranlux(rng_array)
            rng_seed = 1
          END IF
          rnnray0 = rng_array(rng_seed)
          rng_seed = rng_seed + 1
          rnnray0 = rnnray0*pmax
          ibin = 1 + rnnray0*dwi
          ib = i_array(ibin,medium)
          IF (( i_array(ibin+1,medium) .GT. ib )) THEN
7871        CONTINUE
              IF((rnnray0.LT.fcum(ib+1,medium)))GO TO7872
              ib=ib+1
            GO TO 7871
7872        CONTINUE
          END IF
          rnnray0 = (rnnray0 - fcum(ib,medium))*c_array(ib,medium)
          xv = xgrid(ib,medium)*exp(log(1+rnnray0)*b_array(ib,medium))
          IF(((xv .LT. xmax)))GO TO7862
        GO TO 7861
7862    CONTINUE
        xv = xv/e
        costhe = 1 - twice_hc2*xv*xv
        csqthe=costhe*costhe
        IF((( 2*rnnray1 .LT. 1 + csqthe )))GO TO7852
      GO TO 7851
7852  CONTINUE
      sinthe=sqrt(1.0-csqthe)
      return
      end
      subroutine prepare_rayleigh_data(ndat,x,f, ne,emin,emax,pe_array,
     *ncbin,fcum,i_array, b_array,c_array)
      implicit none
      integer*4 ndat
      real*8 x(ndat),  f(ndat)
      integer*4 ne
      real*8 emin,  emax,  pe_array(ne)
      integer*4 ncbin
      real*8 fcum(ndat)
      integer*4 i_array(ncbin)
      real*8 b_array(ndat),  c_array(ndat)
      real*8 zero
      real*8 sum0,a,b,x1,x2,pow_x1,pow_x2,dle,e,xmax, anorm,anorm1,anorm
     *2,w,dw,xold,t,aux
      integer*4 i,j,k,ibin
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      write(*,'(a$)') '      preparing data for Rayleigh sampling ... '
      DO 7881 i=1,ndat
        IF((f(i) .EQ. 0.0))f(i) = zero()
7881  CONTINUE
7882  CONTINUE
      sum0=0
      fcum(1)=0
      DO 7891 i=1,ndat-1
        b = log(f(i+1)/f(i))/log(x(i+1)/x(i))
        b_array(i) = b
        x1 = x(i)
        x2 = x(i+1)
        pow_x1 = x1**(2*b)
        pow_x2 = x2**(2*b)
        sum0=sum0+f(i)*f(i)*(x2*x2*pow_x2-x1*x1*pow_x1)/((1+b)*pow_x1)
        fcum(i+1) = sum0
7891  CONTINUE
7892  CONTINUE
      dle = log(emax/emin)/(ne-1)
      i = 1
      DO 7901 j=1,ne
        e = emin*exp(dle*(j-1))
        xmax = 20.607544d0*2*e/prm
        DO 7911 k=i,ndat-1
          IF((xmax .GE. x(k) .AND. xmax .LT. x(k+1)))GO TO7912
7911    CONTINUE
7912    CONTINUE
        i = k
        b = b_array(i)
        x1 = x(i)
        x2 = xmax
        pow_x1 = x1**(2*b)
        pow_x2 = x2**(2*b)
        pe_array(j) = fcum(i) + f(i)*f(i)*(x2*x2*pow_x2-x1*x1*pow_x1)/((
     *  1+b)*pow_x1)
7901  CONTINUE
7902  CONTINUE
      i_array(ncbin) = i
      anorm = 1d0/sqrt(pe_array(ne))
      anorm1 = 1.005d0/pe_array(ne)
      anorm2 = 1d0/pe_array(ne)
      DO 7921 j=1,ne
        pe_array(j) = pe_array(j)*anorm1
        IF((pe_array(j) .GT. 1))pe_array(j) = 1
7921  CONTINUE
7922  CONTINUE
      DO 7931 j=1,ndat
        f(j) = f(j)*anorm
        fcum(j) = fcum(j)*anorm2
        c_array(j) = (1+b_array(j))/(x(j)*f(j))**2
7931  CONTINUE
7932  CONTINUE
      dw = 1d0/(ncbin-1)
      xold = x(1)
      ibin = 1
      b = b_array(1)
      pow_x1 = x(1)**(2*b)
      i_array(1) = 1
      DO 7941 i=2,ncbin-1
        w = dw
7951    CONTINUE
          x1 = xold
          x2 = x(ibin+1)
          t = x1*x1*x1**(2*b)
          pow_x2 = x2**(2*b)
          aux=f(ibin)*f(ibin)*(x2*x2*pow_x2-t)/((1+b)*pow_x1)
          IF (( aux .GT. w )) THEN
            xold = exp(log(t+w*(1+b)*pow_x1/f(ibin)/f(ibin))/(2+2*b))
            i_array(i) = ibin
            GO TO7952
          END IF
          w = w - aux
          xold = x2
          ibin = ibin+1
          b = b_array(ibin)
          pow_x1 = xold**(2*b)
        GO TO 7951
7952    CONTINUE
7941  CONTINUE
7942  CONTINUE
      DO 7961 j=1,ndat
        b_array(j) = 0.5/(1 + b_array(j))
7961  CONTINUE
7962  CONTINUE
      write(*,'(a /)') 'done'
      return
      end
      real*8 function egs_KN_sigma0(e)
      implicit none
      real*8 e
      real*8 con,ko,c1,c2,c3,eps1,eps2
      data con/0.1274783851/
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      ko = e/prm
      IF (( ko .LT. 0.01 )) THEN
        egs_KN_sigma0 = 8.*con/3.*(1-ko*(2-ko*(5.2-13.3*ko)))/prm
        return
      END IF
      c1 = 1./(ko*ko)
      c2 = 1. - 2*(1+ko)*c1
      c3 = (1+2*ko)*c1
      eps2 = 1
      eps1 = 1./(1+2*ko)
      egs_KN_sigma0 = (c1*(1./eps1-1./eps2)+c2*log(eps2/eps1)+eps2*(c3+0
     *.5*eps2)- eps1*(c3+0.5*eps1))/e*con
      return
      end
      real*8 function egs_KN_sigma1(e)
      implicit none
      real*8 e
      real*8 con,ko,c1,c2,c3,eps1,eps2
      data con/0.1274783851/
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      ko = e/prm
      c1 = 1./(ko*ko)
      c2 = 1. - 2*(1+ko)*c1
      c3 = (1+2*ko)*c1
      eps2 = 1
      eps1 = 1./(1+2*ko)
      egs_KN_sigma1 = c1*(1./eps1-1./eps2)
      egs_KN_sigma1 = egs_KN_sigma1 + log(eps2/eps1)*(c2 - c1) - c2*(eps
     *2-eps1)
      egs_KN_sigma1 = egs_KN_sigma1 + c3*(eps2-eps1)*(1-0.5*(eps1+eps2))
      egs_KN_sigma1 = egs_KN_sigma1 + (eps2-eps1)*(0.5*(eps1+eps2)-(eps1
     **eps1+eps2*eps2+eps1*eps2)/3)
      egs_KN_sigma1 = egs_KN_sigma1*con
      return
      end
      subroutine egsi_get_data(flag,iunit,n,ne,zsorted,pz_sorted,ge1,ge0
     *,data)
      implicit none
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      real*8 eth
      integer*4 flag,iunit,n,ne
      real*8 ge1,ge0,zsorted(*),pz_sorted(*),data(*)
      real*8 etmp(2000),ftmp(2000)
      real*8 gle,sig,p,e
      integer*4 i,j,k,kk,iz,iz_old,ndat,iiz
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      rewind(iunit)
      iz_old = 0
      DO 7971 k=1,n
        data(k) = 0
7971  CONTINUE
7972  CONTINUE
      DO 7981 i=1,ne
        iiz = int(zsorted(i)+0.5)
        DO 7991 iz=iz_old+1,iiz
          read(iunit,*,err=8000) ndat
          IF (( ndat .GT. 2000 )) THEN
            write(i_log,'(/a)') '***************** Error: '
            write(i_log,*) 'Too many input data points. Max. is ',2000
            write(i_log,'(/a)') '***************** Quiting now.'
            call exit(1)
          END IF
          IF (( flag .EQ. 0 .OR. flag .EQ. 3)) THEN
            read(iunit,*,err=8000) (etmp(k),ftmp(k),k=1,ndat)
          ELSE
            read(iunit,*,err=8000) (etmp(k+1),ftmp(k+1), k=1,ndat)
            IF (( flag .EQ. 1 )) THEN
              eth = 2*rm
            ELSE
              eth = 4*rm
            END IF
            ndat = ndat + 1
            DO 8011 k=2,ndat
              ftmp(k) = ftmp(k) - 3*log(1-eth/exp(etmp(k)))
8011        CONTINUE
8012        CONTINUE
            ftmp(1) = ftmp(2)
            etmp(1) = log(eth)
          END IF
7991    CONTINUE
7992    CONTINUE
        iz_old = iiz
        DO 8021 k=1,n
          gle = (k - ge0)/ge1
          e = exp(gle)
          IF (( gle .LT. etmp(1) .OR. gle .GE. etmp(ndat) )) THEN
            IF (( flag .EQ. 0 )) THEN
              write(i_log,'(/a)') '***************** Error: '
              write(i_log,*) 'Energy ',exp(gle), ' is outside the availa
     *ble data range of ', exp(etmp(1)),exp(etmp(ndat))
              write(i_log,'(/a)') '***************** Quiting now.'
              call exit(1)
            ELSE IF((flag .EQ. 1 .OR. flag .EQ. 2)) THEN
              IF (( gle .LT. etmp(1) )) THEN
                sig = 0
              ELSE
                sig = exp(ftmp(ndat))
              END IF
            ELSE
              sig = 0
            END IF
          ELSE
            DO 8031 kk=1,ndat-1
              IF((gle .GE. etmp(kk) .AND. gle .LT. etmp(kk+1)))GO TO8032
8031        CONTINUE
8032        CONTINUE
            IF (( flag .NE. 3)) THEN
              p = (gle - etmp(kk))/(etmp(kk+1) - etmp(kk))
              sig = exp(p*ftmp(kk+1) + (1-p)*ftmp(kk))
            ELSE
              p = (e - exp(etmp(kk)))/(exp(etmp(kk+1)) - exp(etmp(kk)))
              sig = p*exp(ftmp(kk+1)) + (1-p)*exp(ftmp(kk))
            END IF
          END IF
          IF(((flag .EQ. 1 .OR. flag .EQ. 2) .AND. e .GT. eth))sig = sig
     *    *(1-eth/e)**3
          data(k) = data(k) + pz_sorted(i)*sig
8021    CONTINUE
8022    CONTINUE
7981  CONTINUE
7982  CONTINUE
      return
8000  CONTINUE
      write(i_log,'(/a)') '***************** Error: '
      write(i_log,*) 'Error while reading user photon cross sections fro
     *m unit ', iunit
      write(i_log,'(/a)') '***************** Quiting now.'
      call exit(1)
      return
      end
      subroutine egsi_get_shell_data(imed,n,ne,zsorted,pz_sorted,ge1,ge0
     *,data)
      implicit none
      common /egs_io/ file_extensions(20), file_units(20), user_code,  i
     *nput_file,  output_file, pegs_file,  hen_house,  egs_home,  work_d
     *ir,  host_name,  n_parallel,  i_parallel,  first_parallel, n_max_p
     *arallel, n_chunk,  n_files, i_input,  i_log,  i_incoh,  i_nist_dat
     *a,  i_mscat,  i_photo_cs,  i_photo_relax,  xsec_out,  is_batch,  i
     *s_pegsless
      character input_file*256, output_file*256, pegs_file*256, file_ext
     *ensions*10, hen_house*128, egs_home*128, work_dir*128, user_code*6
     *4, host_name*64
      integer*4 n_parallel, i_parallel, first_parallel,n_max_parallel, n
     *_chunk, file_units, n_files,i_input,i_log,i_incoh, i_nist_data,i_m
     *scat,i_photo_cs,i_photo_relax, xsec_out
      logical is_batch,is_pegsless
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      COMMON/MEDIA/  RLC(1),RLDU(1),RHO(1),MSGE(1),MGE(1),MSEKE(1),MEKE(
     *1),MLEKE(1),MCMFP(1),MRANGE(1),IRAYLM(1),IPHOTONUCM(1), MEDIA(24,1
     *), photon_xsections, comp_xsections, photonuc_xsections,eii_xfile,
     *IPHOTONUC,NMED
      CHARACTER*4 MEDIA
      real*8 RLC,  RLDU,  RHO,  apx, upx
      integer*4 MSGE,  MGE,  MSEKE, MEKE,  MLEKE, MCMFP, MRANGE, IRAYLM,
     *  IPHOTONUCM, IPHOTONUC, NMED
      character*16 eii_xfile
      character*16 photon_xsections
      character*16 comp_xsections
      character*16 photonuc_xsections
      common/pe_shell_data/ pe_xsection(500,100,0:16),  pe_elem_prob(500
     *,100,1),   pe_energy(500,100),  pe_zsorted(100,1), pe_be(100,16),
     * pe_nshell(100),  pe_zpos(100),  pe_nge(100),  pe_ne
      real*8 pe_be, pe_energy, pe_xsection, pe_elem_prob
      integer*4 pe_zsorted, pe_nshell, pe_zpos, pe_nge, pe_ne
      integer*4 n,  ne,  ndat
      real*8 ge1,ge0,zsorted(*),pz_sorted(*),data(*)
      real*8 sigma(500),sigmaMedium
      real*4 etmp(2000),ftmp(2000)
      real*4 gle,sig,p
      integer*4 i,j,k,kk,iz,zpos,imed
      DO 8041 k=1,n
        data(k) = 0
8041  CONTINUE
8042  CONTINUE
      DO 8051 k=1,ne
        sigma(k) = 0
8051  CONTINUE
8052  CONTINUE
      DO 8061 i=1,ne
        iz = int(zsorted(i)+0.5)
        zpos = pe_zpos(iz)
        ndat = pe_nge(zpos)
        DO 8071 k=1,ndat
          pe_elem_prob(k,i,imed) = pz_sorted(i)*pe_xsection(k,zpos,0)
          etmp(k) = pe_energy(k,zpos)
          ftmp(k) = log(pe_xsection(k,zpos,0))
8071    CONTINUE
8072    CONTINUE
        DO 8081 k=1,n
          gle = (k - ge0)/ge1
          IF (( gle .LT. etmp(1) .OR. gle .GE. etmp(ndat) )) THEN
            write(i_log,'(/a)') '***************** Error: '
            write(i_log,*) 'egsi_get_shell_data: Energy ',exp(gle), ' is
     * outside the available data range of ', exp(etmp(1)),exp(etmp(ndat
     *      ))
            write(i_log,'(/a)') '***************** Quiting now.'
            call exit(1)
          ELSE
            DO 8091 kk=1,ndat-1
              IF((gle .GE. etmp(kk) .AND. gle .LT. etmp(kk+1)))GO TO8092
8091        CONTINUE
8092        CONTINUE
            p = (gle - etmp(kk))/(etmp(kk+1) - etmp(kk))
            sig = exp(p*ftmp(kk+1) + (1-p)*ftmp(kk))
          END IF
          data(k) = data(k) + pz_sorted(i)*sig
8081    CONTINUE
8082    CONTINUE
8061  CONTINUE
8062  CONTINUE
      DO 8101 i=1,ne
        iz = int(zsorted(i)+0.5)
        zpos = pe_zpos(iz)
        ndat = pe_nge(zpos)
        DO 8111 k=1,ndat
          sig = sigmaMedium(imed,pe_energy(k,zpos))
          pe_elem_prob(k,i,imed) = log(pe_elem_prob(k,i,imed)/sig)
8111    CONTINUE
8112    CONTINUE
8101  CONTINUE
8102  CONTINUE
      return
      end
      real*8 function sigmaMedium(imed, logE)
      implicit none
      COMMON/BREMPR/ DL1(8,1),DL2(8,1),DL3(8,1),DL4(8,1),DL5(8,1),DL6(8,
     *1), ALPHI(2,1),BPAR(2,1),DELPOS(2,1), WA(1,50),PZ(1,50),ZELEM(1,50
     *),RHOZ(1,50), PWR2I(50), DELCM(1),ZBRANG(1),LZBRANG(1),NNE(1), IBR
     *DST,IPRDST,ibr_nist,pair_nrc,itriplet, ASYM(1,50,2)
      CHARACTER*4 ASYM
      real*8 DL1,DL2,DL3,DL4,DL5,DL6,   ALPHI,  BPAR,  DELPOS,  WA,  PZ,
     *  ZELEM,  RHOZ,  PWR2I,  DELCM,  ZBRANG,  LZBRANG
      integer*4 NNE,  IBRDST,  IPRDST,  ibr_nist,  itriplet,  pair_nrc
      common/pe_shell_data/ pe_xsection(500,100,0:16),  pe_elem_prob(500
     *,100,1),   pe_energy(500,100),  pe_zsorted(100,1), pe_be(100,16),
     * pe_nshell(100),  pe_zpos(100),  pe_nge(100),  pe_ne
      real*8 pe_be, pe_energy, pe_xsection, pe_elem_prob
      integer*4 pe_zsorted, pe_nshell, pe_zpos, pe_nge, pe_ne
      real*8 logE, slope, sigma
      integer*4 k,imed,Z,zpos,m,ibsearch
      sigmaMedium = 0
      DO 8121 k=1,nne(imed)
        Z = int( zelem(imed,k) + 0.5 )
        zpos = pe_zpos(Z)
        m = ibsearch(logE,pe_nge(zpos),pe_energy(1,zpos))
        slope = log(pe_xsection(m+1,zpos,0)/pe_xsection(m,zpos,0))
        slope = slope/(pe_energy(m+1,zpos)-pe_energy(m,zpos))
        sigma = log(pe_xsection(m,zpos,0))
        sigma = sigma + slope*(logE - pe_energy(m,zpos))
        sigma = exp(sigma)
        sigmaMedium = sigmaMedium + pz(imed,k)*sigma
8121  CONTINUE
8122  CONTINUE
      return
      end
      subroutine egs_heap_sort(n,rarray,jarray)
      implicit none
      integer*4 n,jarray(*)
      real*8 rarray(*)
      integer*4 i,ir,j,l,ira
      real*8 rra
      DO 8131 i=1,n
        jarray(i)=i
8131  CONTINUE
8132  CONTINUE
      IF((n .LT. 2))return
      l=n/2+1
      ir=n
8141  CONTINUE
        IF ((l .GT. 1)) THEN
          l=l-1
          rra=rarray(l)
          ira=l
        ELSE
          rra=rarray(ir)
          ira=jarray(ir)
          rarray(ir)=rarray(1)
          jarray(ir)=jarray(1)
          ir=ir-1
          IF ((ir .EQ. 1)) THEN
            rarray(1)=rra
            jarray(1)=ira
            return
          END IF
        END IF
        i=l
        j=l+l
8151    CONTINUE
          IF((j .GT. ir))GO TO8152
          IF ((j .LT. ir)) THEN
            IF((rarray(j) .LT. rarray(j+1)))j=j+1
          END IF
          IF ((rra .LT. rarray(j))) THEN
            rarray(i)=rarray(j)
            jarray(i)=jarray(j)
            i=j
            j=j+j
          ELSE
            j=ir+1
          END IF
        GO TO 8151
8152    CONTINUE
        rarray(i)=rra
        jarray(i)=ira
      GO TO 8141
8142  CONTINUE
      return
      end
      SUBROUTINE PHOTONUC
      implicit none
      COMMON/STACK/ E(2),X(2),Y(2),Z(2),U(2),V(2),W(2),DNEAR(2),WT(2),IQ
     *(2),IR(2),LATCH(2), LATCHI,NP,NPold
      DOUBLE PRECISION E
      real*8 X,Y,Z,  U,V,W,  DNEAR,  WT
      integer*4 IQ,  IR,  LATCH,  LATCHI, NP,  NPold
      COMMON/EPCONT/EDEP,EDEP_LOCAL,TSTEP,TUSTEP,USTEP,TVSTEP,VSTEP, RHO
     *F,EOLD,ENEW,EKE,ELKE,GLE,E_RANGE, x_final,y_final,z_final, u_final
     *,v_final,w_final, IDISC,IROLD,IRNEW,IAUSFL(35)
      DOUBLE PRECISION EDEP,  EDEP_LOCAL
      real*8 TSTEP,  TUSTEP,  USTEP,  VSTEP,  TVSTEP,  RHOF,  EOLD,  ENE
     *W,  EKE,  ELKE,  GLE,  E_RANGE, x_final,y_final,z_final,  u_final,
     *v_final,w_final
      integer*4 IDISC,  IROLD,  IRNEW,  IAUSFL
      COMMON/USEFUL/PZERO,PRM,PRMT2,RM,MEDIUM,MEDOLD
      DOUBLE PRECISION PZERO,  PRM,  PRMT2
      real*8 RM
      integer*4 MEDIUM,  MEDOLD
      DATA RM,PRM,PRMT2,PZERO/0.5109989461,0.5109989461,1.0219978922,0.D
     *0/
      npold = np
      edep = pzero
      e(np) = pzero
      wt(np) = 0
      return
      end
