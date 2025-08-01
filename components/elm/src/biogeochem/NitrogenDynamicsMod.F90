module NitrogenDynamicsMod

  !-----------------------------------------------------------------------
  ! !MODULE: NitrogenDynamicsMod
  !
  ! !DESCRIPTION:
  ! Module for mineral nitrogen dynamics (deposition, fixation, leaching)
  ! for coupled carbon-nitrogen code.
  !
  ! !USES:
  use shr_kind_mod        , only : r8 => shr_kind_r8
  use decompMod           , only : bounds_type
  use elm_varcon          , only : dzsoi_decomp, zisoi
  use elm_varctl          , only : use_vertsoilc, use_fan
  use subgridAveMod       , only : p2c, p2c_1d_filter
  use atm2lndType         , only : atm2lnd_type
  use CNStateType         , only : cnstate_type
  use CropType            , only : crop_type
  use ColumnType          , only : col_pp
  use ColumnDataType      , only : col_es, col_ws, col_wf, col_cf, col_ns, col_nf
  use ColumnDataType      , only : nfix_timeconst
  use VegetationType      , only : veg_pp
  use VegetationDataType  , only : veg_cs, veg_ns, veg_nf
  use VegetationPropertiesType  , only : veg_vp
  use elm_varctl          , only : NFIX_PTASE_plant
  use elm_varctl          , only : use_fates
  use ELMFatesInterfaceMod  , only : hlm_fates_interface_type
  use FrictionVelocityType, only : frictionvel_type
  use SoilStateType       , only : soilstate_type
  use FanUpdateMod        , only : fan_eval

  !
  implicit none
  save
  private
  !
  ! !PUBLIC MEMBER FUNCTIONS:
  public :: NitrogenDynamicsInit
  public :: NitrogenDeposition
  public :: NitrogenFixation
  public :: NitrogenLeaching
  public :: NitrogenFert
  public :: CNSoyfix
  public :: readNitrogenDynamicsParams
  public :: NitrogenFixation_balance

  !
  ! !PRIVATE DATA:
  type, public :: CNNDynamicsParamsType
     real(r8) :: sf      ! soluble fraction of mineral N (unitless)
     real(r8) :: sf_no3  ! soluble fraction of NO3 (unitless)
  end type CNNDynamicsParamsType

  type(CNNDynamicsParamsType), public ::  CNNDynamicsParamsInst
  !$acc declare create(CNNDynamicsParamsInst)

  logical, private, parameter :: debug_fan = .false.

  !-----------------------------------------------------------------------

contains

  !-----------------------------------------------------------------------
  subroutine NitrogenDynamicsInit ( )
    !
    ! !DESCRIPTION:
    ! Initialize module variables
    !-----------------------------------------------------------------------

    if (nfix_timeconst .eq. -1.2345_r8) then
       ! If nfix_timeconst is equal to the junk default value, then it
       ! wasn't specified by the user namelist and we need to assign
       ! it the correct default value. If the user specified it in the
       ! name list, we leave it alone.
       nfix_timeconst = 10._r8
    end if

  end subroutine NitrogenDynamicsInit

  !-----------------------------------------------------------------------
  subroutine readNitrogenDynamicsParams ( ncid )
    !
    ! !DESCRIPTION:
    ! Read in parameters
    !
    ! !USES:
    use ncdio_pio   , only : file_desc_t,ncd_io
    use abortutils  , only : endrun
    use shr_log_mod , only : errMsg => shr_log_errMsg
    !
    ! !ARGUMENTS:
    type(file_desc_t),intent(inout) :: ncid   ! pio netCDF file id
    !
    ! !LOCAL VARIABLES:
    character(len=32)  :: subname = 'CNNDynamicsParamsType'
    character(len=100) :: errCode = '-Error reading in parameters file:'
    logical            :: readv ! has variable been read in or not
    real(r8)           :: tempr ! temporary to read in constant
    character(len=100) :: tString ! temp. var for reading
    !-----------------------------------------------------------------------

    call NitrogenDynamicsInit()

    tString='sf_minn'
    call ncd_io(varname=trim(tString),data=tempr, flag='read', ncid=ncid, readvar=readv)
    if ( .not. readv ) call endrun(msg=trim(errCode)//trim(tString)//errMsg(__FILE__, __LINE__))
    CNNDynamicsParamsInst%sf=tempr

    tString='sf_no3'
    call ncd_io(varname=trim(tString),data=tempr, flag='read', ncid=ncid, readvar=readv)
    if ( .not. readv ) call endrun(msg=trim(errCode)//trim(tString)//errMsg(__FILE__, __LINE__))
    CNNDynamicsParamsInst%sf_no3=tempr

  end subroutine readNitrogenDynamicsParams

  !-----------------------------------------------------------------------
  subroutine NitrogenDeposition( bounds, atm2lnd_vars )
    !
    ! !DESCRIPTION:
    ! On the radiation time step, update the nitrogen deposition rate
    ! from atmospheric forcing. For now it is assumed that all the atmospheric
    ! N deposition goes to the soil mineral N pool.
    ! This could be updated later to divide the inputs between mineral N absorbed
    ! directly into the canopy and mineral N entering the soil pool.
    !
    ! !ARGUMENTS:
    type(bounds_type)  , intent(in)  :: bounds 
    type(atm2lnd_type) , intent(in)  :: atm2lnd_vars
    !
    ! !LOCAL VARIABLES:
    integer :: g,c,fc                  ! indices
    integer :: begc, endc
    !-----------------------------------------------------------------------

    associate(&
         forc_ndep     =>  atm2lnd_vars%forc_ndep_grc           , & ! Input:  [real(r8) (:)]  nitrogen deposition rate (gN/m2/s)
         ndep_to_sminn =>  col_nf%ndep_to_sminn   & ! Output: [real(r8) (:)]
         )
      begc = bounds%begc
      endc = bounds%endc

      ! Loop through columns
      ! Note: why loop through all columns? adjusting to filter is nonBFB due to averaging, but ndep_to_sminn is only needed
      ! for active soil columns
      !$acc parallel loop independent gang vector default(present)
      do c = begc, endc 
         g = col_pp%gridcell(c)
         ndep_to_sminn(c) = forc_ndep(g)
      end do

    end associate

  end subroutine NitrogenDeposition

  !-----------------------------------------------------------------------
  subroutine NitrogenFixation(bounds, num_soilc, filter_soilc, dayspyr)
    !
    ! !DESCRIPTION:
    ! On the radiation time step, update the nitrogen fixation rate
    ! as a function of annual total NPP. This rate gets updated once per year.
    ! All N fixation goes to the soil mineral N pool.
    !
    ! !USES:
    use elm_varcon       , only : secspday, spval
    use elm_instMod      , only : alm_fates
    !
    ! !ARGUMENTS:
    type(bounds_type)       , intent(in)    :: bounds
    integer                 , intent(in)    :: num_soilc       ! number of soil columns in filter
    integer                 , intent(in)    :: filter_soilc(:) ! filter for soil columns
    real(r8), intent(in) :: dayspyr               ! days per year
    
    !
    ! !LOCAL VARIABLES:
    integer  :: c,fc                  ! indices
    integer  :: ic                    ! clump index
    integer  :: s                     ! site index (fates only)
    real(r8) :: t                     ! temporary
    real(r8) :: secspyr               ! seconds per yr
    logical  :: do_et_bnf = .false.

    ! Test mutliplier of fixation rate, leave as 1 to use base rates
    real(r8),parameter  :: test_mult = 1.0_r8  

    !-----------------------------------------------------------------------

    associate(&
         cannsum_npp    => col_cf%annsum_npp      , & ! Input:  [real(r8) (:)]  nitrogen deposition rate (gN/m2/s)
         col_lag_npp    => col_cf%lag_npp         , & ! Input: [real(r8) (:)]  (gC/m2/s) lagged net primary production
         qflx_tran_veg  => col_wf%qflx_tran_veg    , & ! col vegetation transpiration (mm H2O/s) (+ = to atm)
         qflx_evap_veg  => col_wf%qflx_evap_veg    , & ! col vegetation evaporation (mm H2O/s) (+ = to atm)
         nfix_to_sminn  => col_nf%nfix_to_sminn   & ! Output: [real(r8) (:)]  symbiotic/asymbiotic N fixation to soil mineral N (gN/m2/s)
         )


      if (do_et_bnf) then
         secspyr = dayspyr * 86400._r8

         do fc = 1, num_soilc
            c =filter_soilc(fc)
            !use the cleveland equation
            t = test_mult*(0.00102_r8*(qflx_evap_veg(c)+qflx_tran_veg(c))+0.0524_r8/secspyr)
            nfix_to_sminn(c) = max(0._r8, t)
         enddo
      else
         if(use_fates)then
            ic = bounds%clump_index
            do fc = 1,num_soilc
               c = filter_soilc(fc)
               s = alm_fates%f2hmap(ic)%hsites(c)
               if ( alm_fates%fates(ic)%bc_out(s)%ema_npp > 0._r8) then
                  ! ema_npp is units: [gC/m^2/year] 
                  t = test_mult*(1.8_r8 * (1._r8 - exp(-0.003_r8 * alm_fates%fates(ic)%bc_out(s)%ema_npp )))/(secspday * dayspyr)
                  nfix_to_sminn(c) = max(0._r8,t)
               else
                  nfix_to_sminn(c) = 0._r8
               endif
            end do
         else
         if ( nfix_timeconst > 0._r8 .and. nfix_timeconst < 500._r8 ) then
            ! use exponential relaxation with time constant nfix_timeconst for NPP - NFIX relation
            ! Loop through columns
            do fc = 1,num_soilc
               c = filter_soilc(fc)

               if (col_lag_npp(c) /= spval) then
                  ! need to put npp in units of gC/m^2/year here first
                  t = test_mult*(1.8_r8 * (1._r8 - exp(-0.003_r8 * col_lag_npp(c)*(secspday * dayspyr))))/(secspday * dayspyr)
                  nfix_to_sminn(c) = max(0._r8,t)
               else
                  nfix_to_sminn(c) = 0._r8
               endif
            end do
         else
            ! use annual-mean values for NPP-NFIX relation
            do fc = 1,num_soilc
               c = filter_soilc(fc)

               t = test_mult*(1.8_r8 * (1._r8 - exp(-0.003_r8 * cannsum_npp(c))))/(secspday * dayspyr)
               nfix_to_sminn(c) = max(0._r8,t)
            end do
         endif
         end if
      endif

    end associate

  end subroutine NitrogenFixation

  !-----------------------------------------------------------------------
  subroutine NitrogenLeaching(num_soilc, filter_soilc, dt )
    !
    ! !DESCRIPTION:
    ! On the radiation time step, update the nitrogen leaching rate
    ! as a function of soluble mineral N and total soil water outflow.
    !
    ! !USES:
    use elm_varpar       , only : nlevdecomp, nlevsoi, nlevgrnd
    !
    ! !ARGUMENTS:
    integer                  , intent(in)    :: num_soilc       ! number of soil columns in filter
    integer                  , intent(in)    :: filter_soilc(:) ! filter for soil columns
    real(r8)                 , intent(in)    :: dt          ! radiation time step (seconds)

    !
    ! !LOCAL VARIABLES:
    integer  :: j,c,fc                                 ! indices
    integer  :: nlevbed				       ! number of layers to bedrock
    real(r8) :: disn_conc                              ! dissolved mineral N concentration (gN/kg water)
    real(r8) :: surface_water(num_soilc) ! liquid water to shallow surface depth (kg water/m2)
    real(r8), parameter :: depth_runoff_Nloss = 0.05   ! (m) depth over which runoff mixes with soil water for N loss to runoff
    real(r8) :: sum_var
    real(r8) :: tot_water(num_soilc)
    !-----------------------------------------------------------------------

    associate(&
        nlev2bed            => col_pp%nlevbed                            , & ! Input:  [integer (:)    ]  number of layers to bedrock
         h2osoi_liq          => col_ws%h2osoi_liq            , & ! Input:  [real(r8) (:,:) ]  liquid water (kg/m2) (new) (-nlevsno+1:nlevgrnd)
         qflx_drain          => col_wf%qflx_drain             , & ! Input:  [real(r8) (:)   ]  sub-surface runoff (mm H2O /s)
         qflx_surf           => col_wf%qflx_surf              , & ! Input:  [real(r8) (:)   ]  surface runoff (mm H2O /s)
         smin_no3_vr         => col_ns%smin_no3_vr        , & ! Input:  [real(r8) (:,:) ]
         sf_no3              =>  CNNDynamicsParamsInst%sf_no3  ,  &  ! Assume that 100% of the soil NO3 is in a soluble form
         smin_no3_leached_vr => col_nf%smin_no3_leached_vr , & ! Output: [real(r8) (:,:) ]  rate of mineral NO3 leaching (gN/m3/s)
         smin_no3_runoff_vr  => col_nf%smin_no3_runoff_vr    & ! Output: [real(r8) (:,:) ]  rate of mineral NO3 loss with runoff (gN/m3/s)
         )

     !$acc enter data create(sum_var, surface_water(:num_soilc), tot_water(:num_soilc) )
     ! for runoff calculation; calculate total water to a given depth
     !$acc parallel loop independent gang worker default(present) private(sum_var,c,nlevbed) present(surface_water(:num_soilc))
     do fc = 1,num_soilc
        c = filter_soilc(fc)
        nlevbed = nlev2bed(c)
        sum_var = 0._r8 
        !$acc loop vector reduction(+:sum_var)
        do j = 1,nlevbed
           if ( zisoi(j) <= depth_runoff_Nloss)  then
              sum_var = sum_var + h2osoi_liq(c,j)
           elseif ( zisoi(j-1) < depth_runoff_Nloss)  then
              sum_var = sum_var + h2osoi_liq(c,j) * ((depth_runoff_Nloss - zisoi(j-1)) / col_pp%dz(c,j))
           end if
        end do
        surface_water(fc) = sum_var 
     end do

      ! calculate the total soil water
      !$acc parallel loop independent gang worker default(present) private(sum_var,c,nlevbed) present(tot_water(:num_soilc)) 
      do fc = 1,num_soilc
         c = filter_soilc(fc)
         nlevbed = col_pp%nlevbed(c)
         sum_var = 0._r8 
         !$acc loop vector reduction(+:sum_var)
         do j = 1,nlevbed
            sum_var = sum_var + col_ws%h2osoi_liq(c,j)
         end do
         tot_water(fc) = sum_var
      end do


     !$acc parallel loop independent gang vector collapse(2) default(present) present(tot_water(:num_soilc),surface_water(:num_soilc)) 
     do j = 1,nlevdecomp
        do fc = 1,num_soilc
           c = filter_soilc(fc)

           if (.not. use_vertsoilc) then
              ! calculate the dissolved mineral N concentration (gN/kg water)
              ! assumes that 10% of mineral nitrogen is soluble
              disn_conc = 0._r8
              if (tot_water(fc) > 0._r8) then
                 disn_conc = (sf_no3 * smin_no3_vr(c,j) )/tot_water(fc)
              end if

              ! calculate the N leaching flux as a function of the dissolved
              ! concentration and the sub-surface drainage flux
              smin_no3_leached_vr(c,j) = disn_conc * qflx_drain(c)
           else
              ! calculate the dissolved mineral N concentration (gN/kg water)
              ! assumes that 10% of mineral nitrogen is soluble
              disn_conc = 0._r8
              if (h2osoi_liq(c,j) > 0._r8) then
                 disn_conc = (sf_no3 * smin_no3_vr(c,j) * col_pp%dz(c,j) )/(h2osoi_liq(c,j) )
              end if
              !
              ! calculate the N leaching flux as a function of the dissolved
              ! concentration and the sub-surface drainage flux
              smin_no3_leached_vr(c,j) = disn_conc * qflx_drain(c) * h2osoi_liq(c,j) / ( tot_water(fc) * col_pp%dz(c,j) )
              !
              ! ensure that leaching rate isn't larger than soil N pool
              smin_no3_leached_vr(c,j) = min(smin_no3_leached_vr(c,j), smin_no3_vr(c,j) / dt )
              !
              ! limit the leaching flux to a positive value
              smin_no3_leached_vr(c,j) = max(smin_no3_leached_vr(c,j), 0._r8)
              !
              !
              ! calculate the N loss from surface runoff, assuming a shallow mixing of surface waters into soil and removal based on runoff
              if ( zisoi(j) <= depth_runoff_Nloss )  then
                 smin_no3_runoff_vr(c,j) = disn_conc * qflx_surf(c) * &
                      h2osoi_liq(c,j) / ( surface_water(fc) * col_pp%dz(c,j) )
              elseif ( zisoi(j-1) < depth_runoff_Nloss )  then
                 smin_no3_runoff_vr(c,j) = disn_conc * qflx_surf(c) * &
                      h2osoi_liq(c,j) * ((depth_runoff_Nloss - zisoi(j-1)) / &
                      col_pp%dz(c,j)) / ( surface_water(fc) * (depth_runoff_Nloss-zisoi(j-1) ))
              else
                 smin_no3_runoff_vr(c,j) = 0._r8
              endif
              !
              ! ensure that runoff rate isn't larger than soil N pool
              smin_no3_runoff_vr(c,j) = min(smin_no3_runoff_vr(c,j), smin_no3_vr(c,j) / dt - smin_no3_leached_vr(c,j))
              !
              ! limit the flux to a positive value
              smin_no3_runoff_vr(c,j) = max(smin_no3_runoff_vr(c,j), 0._r8)


           endif
           ! limit the flux based on current smin_no3 state
           ! only let at most the assumed soluble fraction
           ! of smin_no3 be leached on any given timestep
           smin_no3_leached_vr(c,j) = min(smin_no3_leached_vr(c,j), (sf_no3 * smin_no3_vr(c,j))/dt)

           ! limit the flux to a positive value
           smin_no3_leached_vr(c,j) = max(smin_no3_leached_vr(c,j), 0._r8)

        end do
     end do
     !$acc exit data delete(sum_var,surface_water(:num_soilc), tot_water(:num_soilc) )

   end associate
 end subroutine NitrogenLeaching

  !-----------------------------------------------------------------------
  subroutine NitrogenFert(bounds, num_soilc, filter_soilc, num_pcropp, filter_pcropp)
    !
    ! !DESCRIPTION:
    ! On the radiation time step, update the nitrogen fertilizer for crops
    ! All fertilizer goes into the soil mineral N pool.
    !
    ! !USES:
    use FanUpdateMod, only : fan_to_sminn
    use elm_varctl,   only : fan_to_bgc_crop
    !
    ! !ARGUMENTS:
    type(bounds_type)       , intent(in)    :: bounds
    integer                 , intent(in)    :: num_soilc       ! number of soil columns in filter
    integer                 , intent(in)    :: filter_soilc(:) ! filter for soil columns
    integer , intent(in) :: num_pcropp       ! number of prog. crop patches in filter
    integer , intent(in) :: filter_pcropp(:) ! filter for prognostic crop patches
    !
    ! !LOCAL VARIABLES:
    integer :: c,fc,p,fp                 ! indices
    real(r8) :: manure_col(bounds%begc:bounds%endc)
    !-----------------------------------------------------------------------

    associate(&
         synthfert     =>    veg_nf%synthfert,      & ! Input:  [real(r8) (:)] nitrogen fertilizer rate (gN/m2/s)
         manure        =>    veg_nf%manure,         & ! Input:  [real(r8) (:)] manure nitrogen rate (gN/m2/s)
         totalfert     =>    veg_nf%nfertilization, & ! Input:  [real(r8) (:)] manure nitrogen rate (gN/m2/s) 
         fert_to_sminn =>    col_nf%fert_to_sminn,  & ! Output: [real(r8) (:)]
         begc          =>    bounds%begc,&
         endc          =>    bounds%endc,&
         begp          =>    bounds%begp,&
         endp          =>    bounds%endp &
         )
      if (.not. fan_to_bgc_crop) then
         ! => Crop columns/patches are not handled by FAN. Use synthfert directly and add
         ! the default CLM manure. No N input to non-crop columns in this case.
         ! NOTE: manure_col being allocated over begc:endc is wasteful

        call p2c_1d_filter( bounds,num_soilc,filter_soilc, &
           synthfert(begp:endp),  fert_to_sminn(begc:endc) )

        call p2c_1d_filter( bounds,num_soilc,filter_soilc, &
           manure(begp:endp),  manure_col(begc:endc) )

         ! Add the manure N processed above:
         !$acc parallel loop independent gang vector default(present)
         do fc = 1, num_soilc
            c = filter_soilc(fc)
            fert_to_sminn(c) = fert_to_sminn(c) + manure_col(c)
         end do

         ! Add up synthetic fertilizer and manure to the nfertilization output
         ! variable.
         !$acc parallel loop independent gang vector default(present)
         do fp = 1, num_pcropp
            p = filter_pcropp(fp)
            totalfert(p) = synthfert(p) + manure(p)
         end do
      end if

      ! if fan_to_bgc_crop == .true., FAN fills in the fert_to_sminn and totalfert for
      ! crops. It might also fill in the non-crop columns if enabled.
      call fan_to_sminn(bounds, filter_soilc, num_soilc, totalfert)
    end associate
  end subroutine NitrogenFert

  !-----------------------------------------------------------------------
  subroutine CNSoyfix (bounds, num_soilc, filter_soilc, &
       num_soilp, filter_soilp, &
       crop_vars, cnstate_vars)
    !
    ! !DESCRIPTION:
    ! This routine handles the fixation of nitrogen for soybeans based on
    ! the EPICPHASE model M. Cabelguenne et al., Agricultural systems 60: 175-196, 1999
    ! N-fixation is based on soil moisture, plant growth phase, and availibility of
    ! nitrogen in the soil root zone.
    !
    ! !USES:
    use pftvarcon  , only : nsoybean
    !
    ! !ARGUMENTS:
    type(bounds_type)        , intent(in)    :: bounds 
    integer                  , intent(in)    :: num_soilc       ! number of soil columns in filter
    integer                  , intent(in)    :: filter_soilc(:) ! filter for soil columns
    integer                  , intent(in)    :: num_soilp       ! number of soil patches in filter
    integer                  , intent(in)    :: filter_soilp(:) ! filter for soil patches
    type(crop_type)          , intent(in)    :: crop_vars
    type(cnstate_type)       , intent(in)    :: cnstate_vars
    !
    ! !LOCAL VARIABLES:
    integer :: fp,p,c, begc,endc,begp,endp
    real(r8):: fxw,fxn,fxg,fxr             ! soil water factor, nitrogen factor, growth stage factor
    real(r8):: soy_ndemand                 ! difference between nitrogen supply and demand
    real(r8):: GDDfrac
    real(r8), parameter :: sminnthreshold1= 30._r8, sminnthreshold2= 10._r8
    real(r8),parameter:: GDDfracthreshold1= 0.15_r8, GDDfracthreshold2= 0.30_r8
    real(r8),parameter:: GDDfracthreshold3= 0.55_r8, GDDfracthreshold4= 0.75_r8
    !-----------------------------------------------------------------------

    associate(                                                         &
         wf               =>  col_ws%wf                 , & ! Input:  [real(r8) (:) ]  soil water as frac. of whc for top 0.5 m

         hui              =>  crop_vars%gddplant_patch               , & ! Input:  [real(r8) (:) ]  gdd since planting (gddplant)

         fpg              =>  cnstate_vars%fpg_col                   , & ! Input:  [real(r8) (:) ]  fraction of potential gpp (no units)
         gddmaturity      =>  cnstate_vars%gddmaturity_patch         , & ! Input:  [real(r8) (:) ]  gdd needed to harvest
         croplive         =>  crop_vars%croplive_patch            , & ! Input:  [logical  (:) ]  true if planted and not harvested

         sminn            =>  col_ns%sminn           , & ! Input:  [real(r8) (:) ]  (kgN/m2) soil mineral N
         plant_ndemand    =>  veg_nf%plant_ndemand  , & ! Input:  [real(r8) (:) ]  N flux required to support initial GPP (gN/m2/s)

         soyfixn          =>  veg_nf%soyfixn        , & ! Output: [real(r8) (:) ]  nitrogen fixed to each soybean crop
         soyfixn_to_sminn =>  col_nf%soyfixn_to_sminn   & ! Output: [real(r8) (:) ]
         )

     begc = bounds%begc
     endc = bounds%endc
     begp = bounds%begp
     endp = bounds%endp
      !$acc parallel loop independent gang vector default(present) private(p,c)
      do fp = 1,num_soilp
         p = filter_soilp(fp)
         c = veg_pp%column(p)

         ! if soybean currently growing then calculate fixation

         if (veg_pp%itype(p) == nsoybean .and. croplive(p)) then

            ! difference between supply and demand

            if (fpg(c) < 1._r8) then
               soy_ndemand = 0._r8
               soy_ndemand = plant_ndemand(p) - plant_ndemand(p)*fpg(c)

               ! fixation depends on nitrogen, soil water, and growth stage

               ! soil water factor

               fxw = 0._r8
               fxw = wf(c)/0.85_r8

               ! soil nitrogen factor (Beth says: CHECK UNITS)

               if (sminn(c) > sminnthreshold1) then
                  fxn = 0._r8
               else if (sminn(c) > sminnthreshold2 .and. sminn(c) <= sminnthreshold1) then
                  fxn = 1.5_r8 - .005_r8 * (sminn(c) * 10._r8)
               else if (sminn(c) <= sminnthreshold2) then
                  fxn = 1._r8
               end if

               ! growth stage factor
               ! slevis: to replace GDDfrac, assume...
               ! Beth's crit_offset_gdd_def is similar to my gddmaturity
               ! Beth's ac_gdd (base 5C) similar to my hui=gddplant (base 10
               ! for soy)
               ! Ranges below are not firm. Are they lit. based or tuning based?

               GDDfrac = hui(p) / gddmaturity(p)

               if (GDDfrac <= GDDfracthreshold1) then
                  fxg = 0._r8
               else if (GDDfrac > GDDfracthreshold1 .and. GDDfrac <= GDDfracthreshold2) then
                  fxg = 6.67_r8 * GDDfrac - 1._r8
               else if (GDDfrac > GDDfracthreshold2 .and. GDDfrac <= GDDfracthreshold3) then
                  fxg = 1._r8
               else if (GDDfrac > GDDfracthreshold3 .and. GDDfrac <= GDDfracthreshold4) then
                  fxg = 3.75_r8 - 5._r8 * GDDfrac
               else  ! GDDfrac > GDDfracthreshold4
                  fxg = 0._r8
               end if

               ! calculate the nitrogen fixed by the soybean

               fxr = min(1._r8, fxw, fxn) * fxg
               fxr = max(0._r8, fxr)
               soyfixn(p) =  fxr * soy_ndemand
               soyfixn(p) = min(soyfixn(p), soy_ndemand)

            else ! if nitrogen demand met, no fixation

               soyfixn(p) = 0._r8

            end if

         else ! if not live soybean, no fixation

            soyfixn(p) = 0._r8

         end if
      end do

      call p2c_1d_filter(bounds,num_soilc,filter_soilc, &
        soyfixn(begp:endp), soyfixn_to_sminn(begc:endc))

    end associate

  end subroutine CNSoyfix

  !-----------------------------------------------------------------------
  subroutine NitrogenFixation_balance(num_soilc, filter_soilc,cnstate_vars)
    !
    ! !DESCRIPTION:
    ! created, Aug 2015 by Q. Zhu
    ! On the radiation time step, update the nitrogen fixation rate
    ! as a function of (1) root NP status, (2) fraction of root that is nodulated, (3) carbon cost of root nitrogen uptake
    ! N2 fixation is based on Fisher 2010 GBC doi:10.1029/2009GB003621; Wang 2007 GBC doi:10.1029/2006GB002797; and Grand 2012 ecosys model
    !
    ! !USES:
    use elm_varcon       , only : secspday, spval
    use pftvarcon        , only : noveg

    !
    ! !ARGUMENTS:
    integer                 , intent(in)    :: num_soilc       ! number of soil columns in filter
    integer                 , intent(in)    :: filter_soilc(:) ! filter for soil columns
    type(cnstate_type)      , intent(inout) :: cnstate_vars
    !
    ! !LOCAL VARIABLES:
    integer  :: c,fc,p                     ! indices
    real(r8) :: r_fix                      ! carbon cost of N2 fixation, gC/gN
    real(r8) :: r_nup                      ! carbon cost of root N uptake, gC/gN
    real(r8) :: f_nodule                   ! empirical, fraction of root that is nodulated
    real(r8) :: N2_aq                      ! aqueous N2 bulk concentration gN/m3 soil
    real(r8) :: nfix_tmp
    real(r8) :: sminn_sum, ecosysn_sum
    !-----------------------------------------------------------------------

    associate(&
         ivt                   => veg_pp%itype                         , & ! input:  [integer  (:) ]  pft vegetation type
         cn_scalar             => cnstate_vars%cn_scalar               , &
         cp_scalar             => cnstate_vars%cp_scalar               , &
         vmax_nfix             => veg_vp%vmax_nfix                 , &
         km_nfix               => veg_vp%km_nfix                   , &
         frootc                => veg_cs%frootc        , &
         nfix_to_sminn         => col_nf%nfix_to_sminn  , & ! output: [real(r8) (:)]  symbiotic/asymbiotic n fixation to soil mineral n (gn/m2/s)
         nfix_to_plantn        => veg_nf%nfix_to_plantn , &
         nfix_to_ecosysn       => col_nf%nfix_to_ecosysn, &
         pnup_pfrootc          => veg_ns%pnup_pfrootc, &
         benefit_pgpp_pleafc   => veg_ns%benefit_pgpp_pleafc , &
         t_soi10cm_col         => col_es%t_soi10cm       , &
         h2osoi_vol            => col_ws%h2osoi_vol       , &
         t_scalar              => col_cf%t_scalar           &
         )

      !$acc enter data create(sminn_sum,ecosysn_sum)
      !$acc parallel loop independent gang worker default(present) private(sminn_sum,ecosysn_sum)
      do fc=1,num_soilc
          c = filter_soilc(fc)
          sminn_sum = 0.0_r8
          ecosysn_sum = 0._r8
          !$acc loop vector reduction(+:sminn_sum,ecosysn_sum)
          do p = col_pp%pfti(c), col_pp%pftf(c)
              if (veg_pp%active(p).and. (veg_pp%itype(p) .ne. noveg)) then
                  ! calculate c cost of n2 fixation: fisher 2010 gbc doi:10.1029/2009gb003621
                  r_fix = -6.25_r8*(exp(-3.62_r8 + 0.27_r8*(t_soi10cm_col(c)-273.15_r8)*(1.0_r8-0.5_r8&
                       *(t_soi10cm_col(c)-273.15_r8)/25.15_r8))-2.0_r8)
                  ! calculate c cost of root n uptake: rastetter 2001, ecosystems, 4(4), 369-388.
                  r_nup = benefit_pgpp_pleafc(p) / max(pnup_pfrootc(p),1e-20_r8)
                  ! calculate fraction of root that is nodulated: wang 2007 gbc doi:10.1029/2006gb002797
                  f_nodule = 1 - min(1.0_r8,r_fix / max(r_nup, 1e-20_r8))
                  ! np limitation factor of n2 fixation (not considered now)
                  ! calculate aqueous N2 concentration and bulk aqueous N2 concentration
                  ! aqueous N2 concentration under pure nitrogen is 6.1e-4 mol/L/atm (based on Hery's law)
                  ! 78% atm * 6.1e-4 mol/L/atm * 28 g/mol * 1e3L/m3 * water content m3/m3 at 10 cm
                  N2_aq = 0.78_r8 * 6.1e-4_r8 *28._r8 *1.e3_r8 * max(h2osoi_vol(c,4),0.01_r8)
                  ! calculate n2 fixation rate for each pft and add it to column total
                  nfix_tmp = vmax_nfix(veg_pp%itype(p)) * frootc(p) * cn_scalar(p) *f_nodule * t_scalar(c,1) * &
                             N2_aq/ (N2_aq + km_nfix(veg_pp%itype(p)))
                  if (NFIX_PTASE_plant) then
                     sminn_sum = sminn_sum + nfix_tmp  * veg_pp%wtcol(p) * (1._r8-veg_vp%alpha_nfix(veg_pp%itype(p)))
                     nfix_to_plantn(p) = nfix_tmp * veg_vp%alpha_nfix(veg_pp%itype(p))
                     ecosysn_sum = ecosysn_sum + nfix_tmp  * veg_pp%wtcol(p)
                  else
                     sminn_sum = sminn_sum + nfix_tmp  * veg_pp%wtcol(p)
                     nfix_to_plantn(p) = 0.0_r8
                     ecosysn_sum = ecosysn_sum + nfix_tmp  * veg_pp%wtcol(p)
                  end if
              end if
          end do
          nfix_to_sminn(c) = sminn_sum 
          nfix_to_ecosysn(c) = ecosysn_sum
      end do

    end associate

  end subroutine NitrogenFixation_balance

end module NitrogenDynamicsMod
