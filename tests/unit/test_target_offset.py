"""Tests for the internal target offset."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from homeassistant.components.climate.const import HVACMode
import pytest

from custom_components.better_thermostat.climate import BetterThermostat
from custom_components.better_thermostat.events.trv import convert_outbound_states
from custom_components.better_thermostat.number import (
    BetterThermostatTargetOffsetNumber,
)
from custom_components.better_thermostat.utils.const import (
    CONF_CALIBRATION,
    CONF_CALIBRATION_MODE,
    CalibrationMode,
    CalibrationType,
)


def test_offset_changes_regulation_target_not_user_target():
    """The climate target remains the user's base target while regulation shifts."""
    bt = object.__new__(BetterThermostat)
    bt.bt_target_temp = 19.0
    bt.target_offset = 0.5
    bt.bt_min_temp = 5.0
    bt.bt_max_temp = 30.0

    assert bt.target_temperature == 19.0
    assert bt.regulation_target_temp == 19.5


@pytest.mark.asyncio
async def test_offset_number_updates_live_regulation_value():
    """Changing the number updates the offset and queues control."""
    bt = SimpleNamespace(
        unique_id="entry-1",
        device_name="Test",
        target_offset=0.0,
        bt_target_temp=19.0,
        cooler_entity_id=None,
        startup_running=False,
        bt_hvac_mode=HVACMode.HEAT,
        control_queue_task=SimpleNamespace(put=AsyncMock()),
        async_write_ha_state=Mock(),
    )
    number = BetterThermostatTargetOffsetNumber(bt)
    number.async_write_ha_state = Mock()

    await number.async_set_native_value(0.5)

    assert bt.target_offset == 0.5
    assert bt.bt_target_temp == 19.0
    bt.control_queue_task.put.assert_awaited_once_with(bt)


def test_outbound_setpoint_includes_offset():
    """An uncalibrated TRV receives the effective, offset-adjusted target."""
    entity_id = "climate.trv"
    trv = SimpleNamespace(
        advanced={
            CONF_CALIBRATION: CalibrationType.TARGET_TEMP_BASED,
            CONF_CALIBRATION_MODE: CalibrationMode.NO_CALIBRATION,
        },
        hvac_modes=[HVACMode.HEAT, HVACMode.OFF],
        current_temperature=18.0,
    )
    bt = SimpleNamespace(
        device_name="Test",
        bt_target_temp=19.0,
        target_offset=0.5,
        regulation_target_temp=19.5,
        bt_hvac_mode=HVACMode.HEAT,
        real_trvs={entity_id: trv},
    )

    payload = convert_outbound_states(bt, entity_id, HVACMode.HEAT)

    assert payload["temperature"] == 19.5
    assert bt.bt_target_temp == 19.0
