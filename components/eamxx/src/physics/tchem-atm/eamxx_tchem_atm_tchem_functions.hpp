#ifndef EAMXX_TCHEM_ATM_TCHEM_FUNCTIONS_HPP
#define EAMXX_TCHEM_ATM_TCHEM_FUNCTIONS_HPP

#include <TChem.hpp>
#include <mam4xx/mo_photo.hpp>

#include "share/physics/eamxx_common_physics_functions.hpp"

#include <string>

namespace scream {
namespace tchem {

mam4::mo_photo::PhotoTableData read_photo_table_uci(
    const std::string& rsf_file, const std::string& xs_long_file);

int compute_nsamples(
    const Kokkos::View<const int*>& ntropopause,
    int ncol,
    int nlev,
    bool above);

void compute_offsets(
    const Kokkos::View<const int*>& ntropopause,
    int ncol,
    int nlev,
    const Kokkos::View<int*>& offsets,
    bool above);

void compute_sample_indices(
    const Kokkos::View<const int*>& ntropopause,
    const Kokkos::View<const int*>& offsets,
    int ncol,
    int nlev,
    const Kokkos::View<int*>& sample_icol,
    const Kokkos::View<int*>& sample_ilev,
    bool above);

template <typename StateViewType, typename SourceViewType>
void pack_into_state(
    const StateViewType& state,
    const SourceViewType& src,
    const Kokkos::View<const int*>& sample_icol,
    const Kokkos::View<const int*>& sample_ilev,
    int nsamples,
    int state_col,
    const char* kernel_name) {
  Kokkos::parallel_for(
      kernel_name, Kokkos::RangePolicy<TChem::exec_space>(0, nsamples),
      KOKKOS_LAMBDA(const int isample) {
        state(isample, state_col) = src(sample_icol(isample), sample_ilev(isample));
      });
}

template <typename StateViewType, typename SourceViewType>
void fill_state_column_from_field(
    const StateViewType& state,
    const SourceViewType& src,
    int nlevs,
    int nbatch,
    int state_col,
    const char* kernel_name) {
  Kokkos::parallel_for(
      kernel_name, Kokkos::RangePolicy<TChem::exec_space>(0, nbatch),
      KOKKOS_LAMBDA(const int i) {
        const int icol = i / nlevs;
        const int ilev = i % nlevs;
        state(i, state_col) = src(icol, ilev);
      });
}

template <typename StateViewType, typename TracerViewType, typename QvViewType>
void pack_wet_mmr_into_state(
    const StateViewType& state,
    const TracerViewType& q_tracer,
    const QvViewType& qv,
    const Kokkos::View<const int*>& sample_icol,
    const Kokkos::View<const int*>& sample_ilev,
    int nsamples,
    int state_col,
    Real species_mw,
    const char* kernel_name) {
  using PF = scream::PhysicsFunctions<DefaultDevice>;
  constexpr Real air_mw = scream::physics::Constants<Real>::MWdry.value;

  Kokkos::parallel_for(
      kernel_name, Kokkos::RangePolicy<TChem::exec_space>(0, nsamples),
      KOKKOS_LAMBDA(const int isample) {
        const int icol = sample_icol(isample);
        const int ilev = sample_ilev(isample);
        const Real mmr_wet = q_tracer(icol, ilev);
        const Real qv_wet = qv(icol, ilev);
        const Real mmr_dry = PF::calculate_drymmr_from_wetmmr(mmr_wet, qv_wet);
        state(isample, state_col) = mmr_dry * air_mw / species_mw;
      });
}

template <typename StateViewType, typename TracerViewType, typename QvViewType>
void fill_state_column_from_wet_mmr_field(
    const StateViewType& state,
    const TracerViewType& q_tracer,
    const QvViewType& qv,
    int nlevs,
    int nbatch,
    int state_col,
    Real species_mw,
    const char* kernel_name) {
  using PF = scream::PhysicsFunctions<DefaultDevice>;
  constexpr Real air_mw = scream::physics::Constants<Real>::MWdry.value;
  Kokkos::parallel_for(
      kernel_name, Kokkos::RangePolicy<TChem::exec_space>(0, nbatch),
      KOKKOS_LAMBDA(const int i) {
        const int icol = i / nlevs;
        const int ilev = i % nlevs;
        const Real mmr_wet = q_tracer(icol, ilev);
        const Real qv_wet = qv(icol, ilev);
        const Real mmr_dry = PF::calculate_drymmr_from_wetmmr(mmr_wet, qv_wet);
        state(i, state_col) = mmr_dry * air_mw / species_mw;
      });
}

template <typename DestViewType, typename StateViewType>
void unpack_from_state(
    const DestViewType& dst,
    const StateViewType& state,
    const Kokkos::View<const int*>& sample_icol,
    const Kokkos::View<const int*>& sample_ilev,
    int nsamples,
    int state_col,
    const char* kernel_name) {
  Kokkos::parallel_for(
      kernel_name, Kokkos::RangePolicy<TChem::exec_space>(0, nsamples),
      KOKKOS_LAMBDA(const int isample) {
        const int icol = sample_icol(isample);
        const int ilev = sample_ilev(isample);
        dst(icol, ilev) = state(isample, state_col);
      });
}

template <typename DestViewType, typename SourceViewType>
void pack_photo_rates_into_state(
    const DestViewType& dst,
    const SourceViewType& src,
    const Kokkos::View<const int*>& sample_icol,
    const Kokkos::View<const int*>& sample_ilev,
    int nsamples,
    int nphoto,
    const char* kernel_name) {
  Kokkos::parallel_for(
      kernel_name, Kokkos::RangePolicy<TChem::exec_space>(0, nsamples),
      KOKKOS_LAMBDA(const int isample) {
        const int icol = sample_icol(isample);
        const int ilev = sample_ilev(isample);
        for (int iphoto = 0; iphoto < nphoto; ++iphoto) {
          dst(isample, iphoto) = src(icol, ilev, iphoto);
        }
      });
}

template <typename DestViewType, typename StateViewType, typename QvViewType>
void unpack_wet_mmr_from_state(
    const DestViewType& dst,
    const StateViewType& state,
    const QvViewType& qv,
    const Kokkos::View<const int*>& sample_icol,
    const Kokkos::View<const int*>& sample_ilev,
    int nsamples,
    int state_col,
    Real species_mw,
    const char* kernel_name) {
  using PF = scream::PhysicsFunctions<DefaultDevice>;
  constexpr Real air_mw = scream::physics::Constants<Real>::MWdry.value;

  Kokkos::parallel_for(
      kernel_name, Kokkos::RangePolicy<TChem::exec_space>(0, nsamples),
      KOKKOS_LAMBDA(const int isample) {
        const int icol = sample_icol(isample);
        const int ilev = sample_ilev(isample);
        const Real qv_wet = qv(icol, ilev);
        const Real vmr_dry = state(isample, state_col);
        const Real mmr_dry = vmr_dry * species_mw / air_mw;
        const Real qv_dry = PF::calculate_drymmr_from_wetmmr(qv_wet, qv_wet);
        dst(icol, ilev) = PF::calculate_wetmmr_from_drymmr(mmr_dry, qv_dry);
      });
}

template <typename DestViewType, typename StateViewType>
void fill_field_from_state_column(
    const DestViewType& dst,
    const StateViewType& state,
    int nlevs,
    int nbatch,
    int state_col,
    const char* kernel_name) {
  Kokkos::parallel_for(
      kernel_name, Kokkos::RangePolicy<TChem::exec_space>(0, nbatch),
      KOKKOS_LAMBDA(const int i) {
        const int icol = i / nlevs;
        const int ilev = i % nlevs;
        dst(icol, ilev) = state(i, state_col);
      });
}

template <typename DestViewType, typename StateViewType, typename QvViewType>
void fill_wet_mmr_field_from_state_column(
    const DestViewType& dst,
    const StateViewType& state,
    const QvViewType& qv,
    int nlevs,
    int nbatch,
    int state_col,
    Real species_mw,
    const char* kernel_name) {
  using PF = scream::PhysicsFunctions<DefaultDevice>;
  constexpr Real air_mw = scream::physics::Constants<Real>::MWdry.value;
  Kokkos::parallel_for(
      kernel_name, Kokkos::RangePolicy<TChem::exec_space>(0, nbatch),
      KOKKOS_LAMBDA(const int i) {
        const int icol = i / nlevs;
        const int ilev = i % nlevs;
        const Real qv_wet = qv(icol, ilev);
        const Real vmr_dry = state(i, state_col);
        const Real mmr_dry = vmr_dry * species_mw / air_mw;
        const Real qv_dry = PF::calculate_drymmr_from_wetmmr(qv_wet, qv_wet);
        dst(icol, ilev) = PF::calculate_wetmmr_from_drymmr(mmr_dry, qv_dry);
      });
}

}  // namespace tchem
}  // namespace scream

#endif  // EAMXX_TCHEM_ATM_TCHEM_FUNCTIONS_HPP
