import sc2
from sc2 import Difficulty, Race, maps, run_game, position
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY, OBSERVER, ROBOTICSFACILITY, SCV
from sc2.player import Bot, Computer
import sys
import random
import neural_network
import numpy as np
import pickle
import math


class CustomBot(sc2.BotAI):
    def __init__(self):
        self.SECONDS_PER_MINUTE = 60
        self.MAX_WORKERS = 50

        # {UNIT_TAG: LOCATION}
        self.scouts_and_spots = {}
        self.end_game = False

        self.GLOBAL_TIME = 0
        self.PLAYER_ID = -1
        self.GENERATION = 0
        self.ACTION_VECTOR = [0] * 3
        print(len(sys.argv))

        if len(sys.argv) > 2:
            self.PLAYER_ID = sys.argv[1]
            self.GENERATION = sys.argv[2]

        f = open("python-ai-sc2\\weights\\generation_" + str(self.GENERATION) + 
                 "_weights_" + str(self.PLAYER_ID) + ".pkl", "rb")
        self.PLAYER_BRAIN = pickle.load(f)
        f.close()

    def on_end(self, game_result):
        print('---End of the game---')
        f = open("python-ai-sc2\\results\\generation_" + str(self.GENERATION) +
                 "_player_" + str(self.PLAYER_ID) + ".txt", "w")
        if not self.end_game:
            f.write(str(game_result) + "\n")
        else:
            f.write("Result.Defeat\n")

        f.write(str(self.ACTION_VECTOR) + "\n")

    async def on_step(self, iteration):
        self.GLOBAL_TIME = (self.state.game_loop / 22.4)
        await self.build_scout()
        await self.scout()
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        await self.attack()

    async def build_scout(self):
        if len(self.units(OBSERVER)) < (self.GLOBAL_TIME / 100):
            if len(self.units(OBSERVER)) < 10:
                for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
                    if self.can_afford(OBSERVER) and self.supply_left > 0:
                        await self.do(rf.train(OBSERVER))

    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += random.randrange(-5, 5)
        y += random.randrange(-5, 5)

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x, y)))
        return go_to

    async def scout(self):
        self.expand_dis_dir = {}
        for el in self.expansion_locations:
            distance_to_enemy_start = el.distance_to(
                self.enemy_start_locations[0])
            self.expand_dis_dir[distance_to_enemy_start] = el
        self.ordered_exp_distances = sorted(k for k in self.expand_dis_dir)

        existing_ids = [unit.tag for unit in self.units]
        to_be_removed = []
        for noted_scount in self.scouts_and_spots:
            if noted_scount not in existing_ids:
                to_be_removed.append(noted_scount)

        for scount in to_be_removed:
            del self.scouts_and_spots[scount]

        if len(self.units(ROBOTICSFACILITY).ready) == 0:
            unit_type = PROBE
            unit_limit = 1
        else:
            unit_type = OBSERVER
            unit_limit = 15

        assign_scout = True

        if unit_type == PROBE:
            for unit in self.units(PROBE):
                if unit.tag in self.scouts_and_spots:
                    assign_scout = False

        if assign_scout:
            if len(self.units(unit_type).idle) > 0:
                for obs in self.units(unit_type).idle[:unit_limit]:
                    if obs.tag not in self.scouts_and_spots:
                        for dist in self.ordered_exp_distances:
                            try:
                                location = self.expand_dis_dir[dist]
                                active_locations = [
                                    self.scouts_and_spots[k]
                                    for k in self.scouts_and_spots
                                ]

                                if location not in active_locations:
                                    if unit_type == PROBE:
                                        for unit in self.units(PROBE):
                                            if unit.tag in self.scouts_and_spots:
                                                continue
                                    await self.do(obs.move(location))
                                    self.scouts_and_spots[obs.tag] = location
                                    break
                            except Exception as e:
                                #print(str(e))
                                pass
        for obs in self.units(unit_type):
            if obs.tag in self.scouts_and_spots:
                if obs in [probe for probe in self.units(PROBE)]:
                    await self.do(
                        obs.move(
                            self.random_location_variance(
                                self.scouts_and_spots[obs.tag])))

    async def build_workers(self):
        if len(self.units(NEXUS)) * 16 > len(self.units(PROBE)):
            if len(self.units(PROBE)) < self.MAX_WORKERS:
                for nexus in self.units(NEXUS).ready.noqueue:
                    if self.can_afford(PROBE):
                        await self.do(nexus.train(PROBE))

    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexuses.first)

    async def build_assimilators(self):
        for nexus in self.units(NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0,
                                                           vaspene).exists:
                    await self.do(worker.build(ASSIMILATOR, vaspene))

    async def expand(self):
        if self.units(NEXUS).amount < 3 and self.can_afford(NEXUS):
            await self.expand_now()

    async def offensive_force_buildings(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random

            if self.units(
                    GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
                if self.can_afford(
                        CYBERNETICSCORE
                ) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)

            elif len(self.units(GATEWAY)) < 1:
                if self.can_afford(
                        GATEWAY) and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon)

            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(ROBOTICSFACILITY)) < 1:
                    if self.can_afford(
                            ROBOTICSFACILITY
                    ) and not self.already_pending(ROBOTICSFACILITY):
                        await self.build(ROBOTICSFACILITY, near=pylon)

            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(STARGATE)) < (
                        self.GLOBAL_TIME / self.SECONDS_PER_MINUTE):
                    if self.can_afford(
                            STARGATE) and not self.already_pending(STARGATE):
                        await self.build(STARGATE, near=pylon)

    async def build_offensive_force(self):
        for sg in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left > 0:
                await self.do(sg.train(VOIDRAY))

    async def attack(self):
        target = False

        if self.GLOBAL_TIME % 5 == 0:
            if self.GLOBAL_TIME > 1500:
                self.end_game = True

            if len(self.units(VOIDRAY).noqueue) > 0:
                if not self.end_game:
                    choice = neural_network.predict_solo_output(self.PLAYER_BRAIN, [
                        self.GLOBAL_TIME / 100.0,
                        len(self.units(VOIDRAY).noqueue),
                        len(self.known_enemy_units.exclude_type([SCV])),
                        len(self.known_enemy_units.of_type([SCV]))
                    ])
                else:
                    choice = random.randrange(0, 2)

                self.ACTION_VECTOR[choice] += 1

                if choice == 0:
                    return
                elif choice == 1:
                    if len(self.known_enemy_units) > 0:
                        target = self.known_enemy_units.closest_to(
                            random.choice(self.units(NEXUS)))
                elif choice == 2:
                    if len(self.known_enemy_structures) > 0:
                        target = random.choice(self.known_enemy_structures)
                        for vr in self.units(VOIDRAY).noqueue:
                            await self.do(vr.attack(target))
                    else:
                        target = self.enemy_start_locations[0]

                if target:
                    for vr in self.units(VOIDRAY).idle:
                        await self.do(vr.attack(target))


run_game(
    maps.get("AbyssalReefLE"),
    [Bot(Race.Protoss, CustomBot()),
     Computer(Race.Terran, Difficulty.Medium)],
    realtime=False)
