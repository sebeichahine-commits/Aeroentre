"""
Drone IA de guidage d'évacuation en cas d'incendie (simulation)
----------------------------------------------------------------
Fonctions :
  1. Perception : carte de la zone (murs, feu) + danger thermique autour du feu
  2. Planification : A* avec coût de danger -> chemin sûr vers la meilleure sortie
  3. Guidage : le drone vole devant la victime et lui donne la direction
  4. Blocage : si plus de chemin sûr, l'IA calcule le chemin qui traverse le moins
     de flammes et demande aux pompiers d'éteindre précisément ces cellules
  5. Replanification à chaque pas (le feu se propage)

Pour passer au réel : remplacer
  - Environment  -> flux caméra thermique / LiDAR / SLAM
  - FirefighterLink.send -> radio / MQTT / API du centre de secours
  - Drone.fly_to / Drone.signal -> commandes MAVLink / haut-parleur / LEDs
"""
import heapq
import json
import random
from dataclasses import dataclass

FREE, WALL, FIRE = ".", "#", "F"

MAP = [
    "####################",
    "#........#.........#",
    "#.######.#.#######.#",
    "#.#......#.#.....#.#",
    "#.#.######.#.###.#.#",
    "#.#.#......#.#...#.#",
    "#.#.#.######.#.###.#",
    "#...#........#.....#",
    "#.###.########.###.#",
    "#.....#..........#.#",
    "####.##.#########..#",
    "#.................##",
    "####################",
]
EXITS = [(11, 1), (9, 18)]          # (ligne, colonne)
VICTIM_START = (1, 1)
FIRE_START = [(7, 6), (3, 15)]

HOT_LIMIT = 1.2          # au-delà : trop chaud pour passer en mode "sûr"
DANGER_WEIGHT = 6.0      # poids du danger dans le coût du chemin
SPREAD_PROB = 0.12       # propagation du feu par tick


# ---------------------------------------------------------------- environnement
class Environment:
    def __init__(self):
        self.grid = [list(r) for r in MAP]
        self.h, self.w = len(self.grid), len(self.grid[0])
        for r, c in FIRE_START:
            self.grid[r][c] = FIRE

    def in_bounds(self, p):
        return 0 <= p[0] < self.h and 0 <= p[1] < self.w

    def cell(self, p):
        return self.grid[p[0]][p[1]]

    def neighbors(self, p):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (p[0] + dr, p[1] + dc)
            if self.in_bounds(q) and self.cell(q) != WALL:
                yield q

    def fire_cells(self):
        return [(r, c) for r in range(self.h) for c in range(self.w)
                if self.grid[r][c] == FIRE]

    def spread_fire(self):
        new = []
        for f in self.fire_cells():
            for n in self.neighbors(f):
                if self.cell(n) == FREE and random.random() < SPREAD_PROB:
                    new.append(n)
        for r, c in new:
            self.grid[r][c] = FIRE

    def extinguish(self, cells):
        for r, c in cells:
            if self.grid[r][c] == FIRE:
                self.grid[r][c] = FREE

    def danger_map(self):
        """Danger = chaleur/fumée qui décroît avec la distance au feu."""
        fires = self.fire_cells()
        dm = {}
        for r in range(self.h):
            for c in range(self.w):
                d = 0.0
                for fr, fc in fires:
                    dist = abs(r - fr) + abs(c - fc)
                    if dist <= 4:
                        d += 1.0 / (1 + dist) ** 1.5 * 2
                dm[(r, c)] = d
        return dm


# ------------------------------------------------------------------ planification
class Planner:
    def __init__(self, env):
        self.env = env

    def astar(self, start, exits, danger, allow_fire=False):
        """Retourne (chemin, coût) vers la meilleure sortie, ou (None, inf)."""
        def h(p):
            return min(abs(p[0] - e[0]) + abs(p[1] - e[1]) for e in exits)

        def step_cost(q):
            if self.env.cell(q) == FIRE:
                return 60.0 if allow_fire else None
            if not allow_fire and danger[q] > HOT_LIMIT:
                return None
            return 1.0 + DANGER_WEIGHT * danger[q]

        open_q = [(h(start), 0.0, start)]
        came, best = {start: None}, {start: 0.0}
        while open_q:
            _, g, cur = heapq.heappop(open_q)
            if cur in exits:
                path = []
                while cur is not None:
                    path.append(cur)
                    cur = came[cur]
                return path[::-1], g
            if g > best.get(cur, 1e9):
                continue
            for q in self.env.neighbors(cur):
                sc = step_cost(q)
                if sc is None:
                    continue
                ng = g + sc
                if ng < best.get(q, 1e9):
                    best[q], came[q] = ng, cur
                    heapq.heappush(open_q, (ng + h(q), ng, q))
        return None, float("inf")

    def safe_path(self, start):
        return self.astar(start, EXITS, self.env.danger_map(), allow_fire=False)[0]

    def breach_plan(self, start):
        """Chemin qui traverse le moins de feu : renvoie les cellules à éteindre."""
        dm = self.env.danger_map()
        path, _ = self.astar(start, EXITS, dm, allow_fire=True)
        if not path:
            return None, []
        blocked = [p for p in path if self.env.cell(p) == FIRE]
        if not blocked:
            # pas de flamme sur le chemin, mais zone trop chaude :
            # on demande d'éteindre les foyers proches des cellules brûlantes
            hot = [p for p in path if dm[p] > HOT_LIMIT]
            blocked = sorted({f for f in self.env.fire_cells()
                              if any(abs(f[0] - p[0]) + abs(f[1] - p[1]) <= 2 for p in hot)})
        return path, blocked


# ------------------------------------------------------------------ communication
class FirefighterLink:
    """Simulation de la liaison radio avec les pompiers."""

    def __init__(self):
        self.pending = []          # (tick_arrivée, cellules)

    def send(self, tick, payload):
        print(f"[RADIO -> POMPIERS] {json.dumps(payload, ensure_ascii=False)}")

    def request_extinguish(self, tick, victim_pos, cells, eta=3):
        self.send(tick, {
            "type": "DEMANDE_EXTINCTION",
            "priorite": "CRITIQUE",
            "victime_position": victim_pos,
            "cellules_a_eteindre": cells,
            "motif": "Seul passage vers la sortie bloque par les flammes",
        })
        self.pending.append((tick + eta, cells))
        print(f"[RADIO <- POMPIERS] Recu. Extinction estimee dans {eta} cycles.")

    def poll_done(self, tick):
        done = [c for t, c in self.pending if t <= tick]
        self.pending = [(t, c) for t, c in self.pending if t > tick]
        return [cell for group in done for cell in group]


# ------------------------------------------------------------------------ drone
class Drone:
    def __init__(self, pos):
        self.pos = pos

    @staticmethod
    def direction(a, b):
        dr, dc = b[0] - a[0], b[1] - a[1]
        return {(1, 0): "BAS", (-1, 0): "HAUT", (0, 1): "DROITE", (0, -1): "GAUCHE"}.get((dr, dc), "ICI")

    def signal(self, msg):
        print(f"[DRONE HAUT-PARLEUR] {msg}")

    def fly_to(self, p):
        self.pos = p


# ---------------------------------------------------------------- IA de contrôle
@dataclass
class State:
    GUIDING = "GUIDAGE"
    WAITING = "ATTENTE_POMPIERS"
    DONE = "EVACUE"
    FAILED = "ECHEC"


def render(env, victim, drone):
    out = []
    for r in range(env.h):
        row = ""
        for c in range(env.w):
            p = (r, c)
            row += ("V" if p == victim else "D" if p == drone.pos
                    else "E" if p in EXITS else env.grid[r][c])
        out.append(row)
    print("\n".join(out))


def main(seed=3, max_ticks=80, verbose_map=True):
    random.seed(seed)
    env, planner = Environment(), Planner(None)
    planner.env = env
    link, drone = FirefighterLink(), Drone(VICTIM_START)
    victim, state, requested = VICTIM_START, State.GUIDING, False

    for tick in range(max_ticks):
        env.spread_fire()
        env.extinguish(link.poll_done(tick))      # les pompiers ont éteint
        if state == State.WAITING and not any(env.cell(c) == FIRE for c in (requested or [])):
            drone.signal("Les pompiers ont ouvert le passage. On repart !")
            state, requested = State.GUIDING, False

        if victim in EXITS:
            state = State.DONE
            break
        if env.cell(victim) == FIRE:
            state = State.FAILED
            break

        path = planner.safe_path(victim)
        if path and state == State.GUIDING:
            # le drone se place 2 cases devant la victime
            ahead = path[min(2, len(path) - 1)]
            drone.fly_to(ahead)
            nxt = path[1]
            drone.signal(f"Suivez-moi ! Allez {Drone.direction(victim, nxt)}. "
                         f"Sortie a {len(path) - 1} pas.")
            victim = nxt                           # la victime avance (simulé)
        elif not path and state == State.GUIDING:
            breach, blocked = planner.breach_plan(victim)
            if breach and blocked:
                drone.signal("Passage bloque. Restez ici, les pompiers arrivent.")
                link.request_extinguish(tick, victim, blocked)
                requested, state = blocked, State.WAITING
            else:
                drone.signal("Aucune sortie. Restez bas, loin des flammes.")
                link.send(tick, {"type": "SOS", "victime_position": victim,
                                 "motif": "Aucune route de sortie"})
                state = State.FAILED
                break
        elif state == State.WAITING:
            drone.fly_to(victim)
            drone.signal("Patientez, restez a distance des flammes.")

        if verbose_map:
            print(f"--- tick {tick} | état {state} ---")
            render(env, victim, drone)

    print(f"\n=== RESULTAT : {state} ===")


if _name_ == "_main_":
    main()