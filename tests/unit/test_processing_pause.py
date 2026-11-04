"""Pause blocks heating learning while preserving passive heat-loss observations."""

import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

from homeassistant.components.climate.const import HVACAction, HVACMode
from homeassistant.util import dt as dt_util
import pytest

from custom_components.better_thermostat.climate import BetterThermostat
from custom_components.better_thermostat.events.trv import convert_outbound_states
from custom_components.better_thermostat.trv import Trv
from custom_components.better_thermostat.utils.calibration.mpc_v2.controller import (
    MpcV2Controller,
)
from custom_components.better_thermostat.utils.calibration.mpc_v2.params import (
    MpcV2Params,
)
from custom_components.better_thermostat.utils.calibration.mpc_v2.state import (
    MpcV2State,
)
from custom_components.better_thermostat.utils.const import (
    CalibrationMode,
    CalibrationType,
)
from custom_components.better_thermostat.utils.controlling import _get_valve_control
from custom_components.better_thermostat.utils.helpers import is_processing_paused
from custom_components.better_thermostat.utils.hvac_action import ToleranceHysteresis
from custom_components.better_thermostat.utils.state_manager import StateManager
from custom_components.better_thermostat.utils.thermal_learning import (
    HeatingPowerTracker,
    HeatLossTracker,
)


@pytest.mark.parametrize(
    ("signal", "paused"),
    [
        (None, True),
        ("on", True),
        ("unknown", True),
        ("unavailable", True),
        ("off", False),
        ("false", False),
        ("closed", False),
    ],
)
def test_pause_signal(hass, signal, paused):
    entity_id = "binary_sensor.pause"
    if signal is not None:
        hass.states.async_set(entity_id, signal)
    bt = SimpleNamespace(hass=hass, pause_sensor_entity_id=entity_id)
    assert is_processing_paused(bt) is paused


def test_no_pause_signal_is_backwards_compatible():
    assert not is_processing_paused(SimpleNamespace())


@pytest.mark.parametrize("cached_action", [HVACAction.HEATING, HVACAction.COOLING])
@pytest.mark.parametrize("mode", [HVACMode.HEAT, HVACMode.HEAT_COOL, HVACMode.OFF])
def test_pause_action_is_immediate_without_changing_mode(hass, cached_action, mode):
    hass.states.async_set("binary_sensor.pause", "on")
    bt = SimpleNamespace(
        hass=hass,
        pause_sensor_entity_id="binary_sensor.pause",
        bt_hvac_mode=mode,
        attr_hvac_action=cached_action,
    )

    assert BetterThermostat.hvac_action.fget(bt) == (
        HVACAction.OFF if mode == HVACMode.OFF else HVACAction.IDLE
    )
    assert bt.bt_hvac_mode == mode
    assert bt.attr_hvac_action == cached_action


async def test_pause_resume_keeps_learning_and_user_intent(hass):
    manager = StateManager(hass, "pause-test")
    pid = manager.get_pid("room")
    pid.pid_kp = 42.0
    pid.pid_integral = 3.0
    pid.pid_last_time = 100.0
    mpc = manager.get_mpc("room")
    mpc.gain_est = 0.4
    mpc.loss_est = 0.2
    mpc.last_learn_temp = 20.0
    mpc.u_integral = 123.0
    controller = MpcV2Controller(MpcV2Params())
    manager.set_mpc_v2_live("room", MpcV2State(controller=controller))
    bt = SimpleNamespace(
        hass=hass,
        pause_sensor_entity_id="binary_sensor.pause",
        bt_hvac_mode=HVACMode.HEAT,
        bt_target_temp=21.0,
        _heating_tracker=HeatingPowerTracker(heating_power=0.25, start_temp=19.0),
        _loss_tracker=HeatLossTracker(heat_loss_rate=0.08, start_temp=20.0),
        state_mgr=manager,
        control_queue_task=asyncio.Queue(maxsize=1),
        async_write_ha_state=Mock(),
    )
    hass.states.async_set(bt.pause_sensor_entity_id, "on")
    await BetterThermostat._trigger_pause_change(bt, None)
    assert bt.control_queue_task.qsize() == 1
    assert bt._heating_tracker.start_temp is None
    assert bt._loss_tracker.start_temp == 20.0
    assert pid.pid_last_time == 0.0
    assert mpc.last_learn_temp is None
    assert mpc.u_integral == 0.0

    hass.states.async_set(bt.pause_sensor_entity_id, "off")
    await BetterThermostat._trigger_pause_change(bt, None)
    assert not is_processing_paused(bt)
    assert bt.control_queue_task.qsize() == 1
    assert (bt.bt_hvac_mode, bt.bt_target_temp) == (HVACMode.HEAT, 21.0)
    assert bt._heating_tracker.heating_power == 0.25
    assert bt._loss_tracker.heat_loss_rate == 0.08
    assert bt._loss_tracker.start_temp == 20.0
    assert (pid.pid_kp, pid.pid_integral) == (42.0, 3.0)
    assert (mpc.gain_est, mpc.loss_est) == (0.4, 0.2)
    assert manager.get_mpc_v2_live("room", MpcV2Params()).controller is controller


async def test_paused_heating_learning_entry_point_is_inert(hass):
    hass.states.async_set("binary_sensor.pause", "on")
    bt = SimpleNamespace(
        hass=hass,
        pause_sensor_entity_id="binary_sensor.pause",
        cur_temp=20.0,
        _heating_tracker=Mock(),
        _loss_tracker=Mock(),
    )
    await BetterThermostat.calculate_heating_power(bt)
    bt._heating_tracker.update.assert_not_called()


async def test_passive_heat_loss_is_learned_across_pause_and_resume(hass):
    bt = SimpleNamespace(
        hass=hass,
        pause_sensor_entity_id="binary_sensor.pause",
        cur_temp=21.0,
        bt_target_temp=22.0,
        bt_target_cooltemp=None,
        bt_hvac_mode=HVACMode.HEAT,
        hvac_mode=HVACMode.HEAT,
        contact_open=False,
        tolerance=0.0,
        ignore_states=False,
        device_name="Test",
        _hysteresis=ToleranceHysteresis(),
        _build_trv_snapshots=Mock(return_value=[]),
        _cooler_previously_active=Mock(return_value=False),
        _heating_tracker=HeatingPowerTracker(heating_power=0.25),
        _loss_tracker=HeatLossTracker(heat_loss_rate=0.01),
        state_mgr=None,
        control_queue_task=asyncio.Queue(maxsize=1),
        async_write_ha_state=Mock(),
        schedule_save_state=Mock(),
    )
    bt._compute_hvac_action = lambda: (
        BetterThermostat._compute_hvac_action_pure(bt).action
    )
    start = dt_util.utcnow()
    hass.states.async_set(bt.pause_sensor_entity_id, "on")
    await BetterThermostat._trigger_pause_change(bt, None)
    assert bt._compute_hvac_action() == HVACAction.IDLE
    with patch(
        "custom_components.better_thermostat.climate.dt_util.utcnow", return_value=start
    ):
        await BetterThermostat.calculate_heat_loss(bt)

    bt.cur_temp = 20.0
    with patch(
        "custom_components.better_thermostat.climate.dt_util.utcnow",
        return_value=start + timedelta(minutes=20),
    ):
        await BetterThermostat.calculate_heat_loss(bt)
    assert bt._loss_tracker.start_ts == start
    assert not bt._loss_tracker.cycles

    hass.states.async_set(bt.pause_sensor_entity_id, "off")
    await BetterThermostat._trigger_pause_change(bt, None)
    assert bt._loss_tracker.start_ts == start
    assert bt._compute_hvac_action() == HVACAction.HEATING
    with patch(
        "custom_components.better_thermostat.climate.dt_util.utcnow",
        return_value=start + timedelta(minutes=21),
    ):
        await BetterThermostat.calculate_heat_loss(bt)

    assert len(bt._loss_tracker.cycles) == 1
    assert bt._loss_tracker.cycles[0]["rate"] == pytest.approx(0.05)
    assert bt._loss_tracker.heat_loss_rate > 0.01
    assert bt._heating_tracker.heating_power == 0.25
    bt.schedule_save_state.assert_called_once()


@pytest.mark.parametrize(
    ("contact_open", "action"), [(True, HVACAction.IDLE), (False, HVACAction.COOLING)]
)
async def test_pause_does_not_enable_invalid_heat_loss_samples(
    hass, contact_open, action
):
    hass.states.async_set("binary_sensor.pause", "on")
    tracker = HeatLossTracker(start_temp=21.0)
    bt = SimpleNamespace(
        hass=hass,
        pause_sensor_entity_id="binary_sensor.pause",
        cur_temp=20.0,
        contact_open=contact_open,
        _compute_hvac_action=Mock(return_value=action),
        _loss_tracker=tracker,
    )
    await BetterThermostat.calculate_heat_loss(bt)
    assert tracker.start_temp is None
    assert not tracker.cycles
    assert tracker.heat_loss_rate == 0.01


@pytest.mark.parametrize("reported_mode", [HVACMode.COOL, HVACMode.HEAT_COOL])
async def test_paused_cooler_reports_exclude_active_cooling_and_settling(
    hass, reported_mode
):
    hass.states.async_set("binary_sensor.pause", "on")
    hass.states.async_set(
        "climate.cooler", reported_mode, {"hvac_action": HVACAction.COOLING}
    )
    tracker = HeatLossTracker(heat_loss_rate=0.01, start_temp=25.0)
    bt = SimpleNamespace(
        hass=hass,
        pause_sensor_entity_id="binary_sensor.pause",
        cooler_entity_id="climate.cooler",
        cur_temp=24.0,
        contact_open=False,
        _compute_hvac_action=Mock(return_value=HVACAction.IDLE),
        _loss_tracker=tracker,
        async_write_ha_state=Mock(),
        schedule_save_state=Mock(),
    )
    start = dt_util.utcnow()
    with patch(
        "custom_components.better_thermostat.climate.dt_util.utcnow", return_value=start
    ):
        await BetterThermostat.calculate_heat_loss(bt)
    assert tracker.start_temp is None
    assert tracker._prev_action == HVACAction.COOLING

    hass.states.async_set("climate.cooler", HVACMode.OFF)
    for elapsed in (1, 5, 10):
        bt.cur_temp -= 0.1
        with patch(
            "custom_components.better_thermostat.climate.dt_util.utcnow",
            return_value=start + timedelta(minutes=elapsed),
        ):
            await BetterThermostat.calculate_heat_loss(bt)
        assert tracker.start_temp is None

    with patch(
        "custom_components.better_thermostat.climate.dt_util.utcnow",
        return_value=start + timedelta(minutes=12),
    ):
        await BetterThermostat.calculate_heat_loss(bt)
    assert tracker.start_temp == bt.cur_temp
    assert tracker.start_ts == start + timedelta(minutes=12)
    assert tracker.heat_loss_rate == 0.01
    assert not tracker.cycles


@pytest.mark.parametrize("calibration_type", list(CalibrationType))
@pytest.mark.parametrize("calibration_mode", list(CalibrationMode))
def test_pause_bypasses_all_calibration_algorithms(
    hass, calibration_type, calibration_mode
):
    hass.states.async_set("binary_sensor.pause", "on")
    trv = Trv.from_legacy_dict(
        "climate.trv",
        {
            "hvac_modes": [HVACMode.HEAT, HVACMode.OFF],
            "min_temp": 5.0,
            "current_temperature": 19.0,
            "advanced": {
                "calibration": calibration_type,
                "calibration_mode": calibration_mode,
            },
        },
    )
    bt = SimpleNamespace(
        hass=hass,
        pause_sensor_entity_id="binary_sensor.pause",
        device_name="Test",
        bt_target_temp=21.0,
        real_trvs={"climate.trv": trv},
    )
    with (
        patch(
            "custom_components.better_thermostat.events.trv.calculate_calibration_local"
        ) as local,
        patch(
            "custom_components.better_thermostat.events.trv.calculate_calibration_setpoint"
        ) as setpoint,
    ):
        payload = convert_outbound_states(bt, "climate.trv", HVACMode.OFF)
    local.assert_not_called()
    setpoint.assert_not_called()
    assert payload["system_mode"] == HVACMode.OFF


def test_pause_closes_direct_valve_even_without_a_valve_algorithm(hass):
    hass.states.async_set("binary_sensor.pause", "on")
    bt = SimpleNamespace(hass=hass, pause_sensor_entity_id="binary_sensor.pause")
    settings, source = _get_valve_control(
        bt,
        "climate.trv",
        CalibrationMode.NO_CALIBRATION,
        CalibrationType.DIRECT_VALVE_BASED,
        hvac_mode=HVACMode.OFF,
    )
    assert settings == {"valve_percent": 0, "apply_valve": True}
    assert source == "hvac_off"
