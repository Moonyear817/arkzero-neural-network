"""Chapter-eight friendly frost/altar pulses from pinned x-1 range data."""

def apply_cold(sim, enemy, duration=10):
    now=sim.current_time
    if enemy.cold_until>now and not enemy.data.freeze_immune:
        enemy.frozen_until=now+duration
        sim._log('FROZEN',unit=enemy.id,until=enemy.frozen_until)
    enemy.cold_until=now+duration
    sim._log('COLD',unit=enemy.id,until=enemy.cold_until)
    sim.state.queue.push(now+duration,'MODIFIER_BOUNDARY',enemy.id)


def pulse(sim, key):
    device=sim.state.devices[key]
    if not device.alive:return
    device.sp=0
    for enemy in sorted(sim.state.enemies.values(),key=lambda u:u.id):
        if not enemy.alive or enemy.hidden or enemy.data.faction!='HOSTILE':continue
        dx,dy=enemy.position[0]-device.position[0],enemy.position[1]-device.position[1]
        if not any(abs(dx-x)<=.5 and abs(dy-y)<=.5 for x in range(-2,3) for y in range(-2,3) if abs(x)+abs(y)<=2):continue
        if device.data.kind=='friendly_frost':apply_cold(sim,enemy)
        elif device.data.kind=='friendly_altar':
            from arknights_sim.combat.runtime import deal_damage
            deal_damage(sim,enemy,2000,'TRUE',event_kind='ALTAR_DAMAGE',defense_sp=False)
        else:raise ValueError('Unsupported pulse device')
    sim.state.mechanic_revision+=1
    sim._log('DEVICE_PULSE',device=key)
    sim.state.queue.push(sim.current_time+device.data.charge,'DEVICE_PULSE',key)
