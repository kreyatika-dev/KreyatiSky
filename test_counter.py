"""
Tests unitaires et fonctionnels pour KreyatiSky TrafficVision
"""
from core.counter import VehicleCounter, intersect, ccw


def test_ccw_and_intersect():
    # Deux segments qui se croisent
    A = (0, 0)
    B = (10, 10)
    C = (0, 10)
    D = (10, 0)
    assert intersect(A, B, C, D) is True

    # Deux segments parallèles qui ne se croisent pas
    A2 = (0, 0)
    B2 = (10, 0)
    C2 = (0, 5)
    D2 = (10, 5)
    assert intersect(A2, B2, C2, D2) is False


def test_vehicle_counter_direction_up_and_down():
    counter = VehicleCounter()
    line_start = (0, 200)
    line_end = (400, 200)

    # 1. Simuler un véhicule montant (UP): Y passe de 220 à 180 (dy < 0)
    counter.update_tracks([{'id': 1, 'bbox': [100, 230, 140, 250], 'class': 'car', 'conf': 0.9}], line_start, line_end)
    # Frame 1: bas de la ligne
    det_f1 = [{'id': 1, 'bbox': [100, 210, 140, 230], 'class': 'car', 'conf': 0.9}]
    counter.update_tracks(det_f1, line_start, line_end)
    assert counter.count_up == 0
    assert counter.count_down == 0

    # Frame 2: haut de la ligne (franchissement)
    det_f2 = [{'id': 1, 'bbox': [100, 170, 140, 190], 'class': 'car', 'conf': 0.9}]
    counter.update_tracks(det_f2, line_start, line_end)
    counter.update_tracks([{'id': 1, 'bbox': [100, 150, 140, 170], 'class': 'car', 'conf': 0.9}], line_start, line_end)
    counter.update_tracks([{'id': 1, 'bbox': [100, 130, 140, 150], 'class': 'car', 'conf': 0.9}], line_start, line_end)
    assert counter.count_up == 1
    assert counter.count_down == 0
    assert counter.get_stats()['total'] == 1
    assert counter.counts_by_class['car']['up'] == 1

    # 2. Simuler un véhicule descendant (DOWN): Y passe de 180 à 220 (dy > 0)
    # Frame 1: haut de la ligne
    counter.update_tracks([{'id': 2, 'bbox': [200, 140, 250, 160], 'class': 'truck', 'conf': 0.85}], line_start, line_end)
    det_down_f1 = [{'id': 2, 'bbox': [200, 170, 250, 190], 'class': 'truck', 'conf': 0.85}]
    counter.update_tracks(det_down_f1, line_start, line_end)
    # Frame 2: franchit vers le bas
    det_down_f2 = [{'id': 2, 'bbox': [200, 210, 250, 230], 'class': 'truck', 'conf': 0.85}]
    counter.update_tracks(det_down_f2, line_start, line_end)
    counter.update_tracks([{'id': 2, 'bbox': [200, 230, 250, 250], 'class': 'truck', 'conf': 0.85}], line_start, line_end)
    counter.update_tracks([{'id': 2, 'bbox': [200, 250, 250, 270], 'class': 'truck', 'conf': 0.85}], line_start, line_end)

    assert counter.count_up == 1
    assert counter.count_down == 1
    assert counter.get_stats()['total'] == 2
    assert counter.counts_by_class['truck']['down'] == 1
    assert counter.count_down == 1


def test_slow_inference_does_not_expire_tracks_and_video_flow_uses_media_time():
    counter = VehicleCounter()
    line_start = (0, 200)
    line_end = (400, 200)

    counter.update_tracks(
        [{'id': 18, 'bbox': [100, 230, 140, 250], 'class': 'car', 'conf': 0.9}],
        line_start, line_end, timeline=9.96,
    )
    counter.update_tracks(
        [{'id': 18, 'bbox': [100, 210, 140, 230], 'class': 'car', 'conf': 0.9}],
        line_start, line_end, timeline=10.0,
    )
    # Des secondes de calcul mural ne doivent pas intervenir : seules les trames comptent.
    counter.update_tracks(
        [{'id': 18, 'bbox': [100, 170, 140, 190], 'class': 'car', 'conf': 0.9}],
        line_start, line_end, timeline=10.04,
    )
    counter.update_tracks(
        [{'id': 18, 'bbox': [100, 150, 140, 170], 'class': 'car', 'conf': 0.9}],
        line_start, line_end, timeline=10.08,
    )
    counter.update_tracks(
        [{'id': 18, 'bbox': [100, 130, 140, 150], 'class': 'car', 'conf': 0.9}],
        line_start, line_end, timeline=10.12,
    )
    assert counter.get_stats()['total'] == 1
    assert counter.get_stats()['flow_rate_vpm'] == 1.0

    # Sur la timeline vidéo, le passage sort de la fenêtre après 60 secondes de média.
    counter.update_tracks([], line_start, line_end, timeline=71.0)
    assert counter.get_stats()['flow_rate_vpm'] == 0.0


def test_detection_is_counted_before_direction():
    counter = VehicleCounter()
    line_start = (0, 200)
    line_end = (400, 200)
    above = [{'id': 7, 'bbox': [100, 150, 140, 190], 'class': 'car', 'conf': 0.9}]

    counter.update_tracks(above, line_start, line_end, timeline=1.0, confirm_frames=2)
    assert counter.get_stats()['total'] == 0
    counter.update_tracks(above, line_start, line_end, timeline=1.04, confirm_frames=2)
    assert counter.get_stats()['total'] == 0  # deux images, mais trop bref
    counter.update_tracks(above, line_start, line_end, timeline=1.7, confirm_frames=2)
    stats = counter.get_stats()
    assert stats['total'] == 1
    assert stats['count_pending'] == 1
    assert stats['count_up'] == 0 and stats['count_down'] == 0
    assert stats['recent_events'][0]['direction'] is None

    below = [{'id': 7, 'bbox': [100, 210, 140, 250], 'class': 'car', 'conf': 0.9}]
    counter.update_tracks(below, line_start, line_end, timeline=1.74, confirm_frames=2)
    counter.update_tracks([{'id': 7, 'bbox': [100, 240, 140, 280], 'class': 'car', 'conf': 0.9}], line_start, line_end, timeline=1.78, confirm_frames=2)
    counter.update_tracks([{'id': 7, 'bbox': [100, 270, 140, 310], 'class': 'car', 'conf': 0.9}], line_start, line_end, timeline=1.82, confirm_frames=2)
    stats = counter.get_stats()
    assert stats['total'] == 1  # la direction ne doit pas incrémenter le total une seconde fois
    assert stats['count_pending'] == 0
    assert stats['count_down'] == 1
    assert stats['recent_events'][0]['direction'] == 'down'


def test_speed_between_two_calibrated_lines():
    counter = VehicleCounter()
    line_a_start, line_a_end = (0, 100), (400, 100)
    line_b_start, line_b_end = (0, 200), (400, 200)

    def frame(y, timeline):
        counter.update_tracks(
            [{'id': 21, 'bbox': [100, y - 10, 140, y + 10], 'class': 'car', 'conf': 0.9}],
            line_a_start, line_a_end, timeline=timeline, confirm_frames=1,
            speed_line_start=line_b_start, speed_line_end=line_b_end,
            speed_distance_m=10.0,
        )

    frame(80, 0.0)
    frame(120, 1.0)
    frame(160, 2.0)
    frame(200, 3.0)  # 40 px/s sur une zone de 100 px = 4 m/s = 14,4 km/h
    stats = counter.get_stats()
    assert stats['max_speed_kmh'] == 14.4
    assert stats['avg_speed_kmh'] == 14.4
    assert stats['speed_vehicles_count'] == 1
    assert stats['recent_events'][0]['max_speed_kmh'] == 14.4


def test_stable_direction_before_line_survives_immediate_disappearance():
    counter = VehicleCounter()
    line_start, line_end = (0, 200), (400, 200)
    for index, y in enumerate((260, 240, 220, 190)):
        counter.update_tracks(
            [{'id': 42, 'bbox': [100, y - 10, 140, y + 10], 'class': 'car', 'conf': 0.9}],
            line_start, line_end, timeline=index * 0.04, confirm_frames=2,
        )
    stats = counter.get_stats()
    assert stats['count_up'] == 1
    assert stats['count_down'] == 0
    assert stats['count_awaiting_direction'] == 0
    assert stats['recent_events'][0]['status'] == 'directed'


def test_pending_statistics_distinguish_before_line_and_uncertain_direction():
    counter = VehicleCounter()
    line_start, line_end = (0, 200), (400, 200)
    stationary = [{'id': 1, 'bbox': [100, 240, 140, 260], 'class': 'car', 'conf': 0.9}]
    counter.update_tracks(stationary, line_start, line_end, timeline=0.0, confirm_frames=1)
    counter.update_tracks(stationary, line_start, line_end, timeline=0.7, confirm_frames=1)
    assert counter.get_stats()['count_awaiting_crossing'] == 1

    # Franchissement avec une trajectoire trop courte pour déterminer le sens.
    counter.update_tracks(
        [{'id': 1, 'bbox': [100, 180, 140, 200], 'class': 'car', 'conf': 0.9}],
        line_start, line_end, timeline=0.74, confirm_frames=1,
    )
    stats = counter.get_stats()
    assert stats['count_awaiting_crossing'] == 0
    assert stats['count_awaiting_direction'] == 1


def test_finished_vehicles_leave_memory_but_keep_statistics():
    counter = VehicleCounter()
    line_a_start, line_a_end = (0, 100), (400, 100)
    line_b_start, line_b_end = (0, 200), (400, 200)
    timeline = 0.0
    # 50 véhicules montants (vitesse mesurée), puis 50 qui restent avant la ligne
    for vehicle_id in range(1, 101):
        ys = range(230, 60, -20) if vehicle_id <= 50 else (300, 300, 300)
        for y in ys:
            timeline += 1.0
            counter.update_tracks(
                [{'id': vehicle_id, 'bbox': [100, y - 10, 140, y + 10], 'class': 'car', 'conf': 0.9}],
                line_a_start, line_a_end, timeline=timeline, confirm_frames=1,
                speed_line_start=line_b_start, speed_line_end=line_b_end,
                speed_distance_m=10.0,
            )
    for _ in range(100):  # Plus aucun véhicule visible : les trajectoires expirent
        timeline += 1.0
        counter.update_tracks([], line_a_start, line_a_end, timeline=timeline)

    stats = counter.get_stats()
    assert len(counter.detected_ids) == 0
    assert len(counter.vehicle_speeds) == 0
    assert len(counter.counted_ids) == 0  # nettoyage en trames, pas en secondes
    assert stats['total'] == 100
    assert stats['count_up'] == 50
    assert stats['count_awaiting_crossing'] == 50
    assert stats['direction_known_count'] == 50
    assert stats['speed_vehicles_count'] == 50
    assert stats['max_speed_kmh'] == stats['avg_speed_kmh'] == 7.2

    counter.reset()
    assert counter.get_stats()['total'] == 0


def test_speed_with_perspective_correction():
    from core.config import AppConfig, ground_distance_fn
    # Perspective marquée : Z(v) = (-40 v + 40) / (2 v + 0,2) mètres
    cfg = AppConfig(source_path="cam.mp4", speed_ground_map=[-40.0, 40.0, 2.0, 0.2],
                    speed_ground_map_source="cam.mp4")
    ground = ground_distance_fn(cfg, 720)
    assert ground_distance_fn(cfg.model_copy(update={"source_path": "autre.mp4"}), 720) is None

    def row_for(z):  # inverse de Z(v), en pixels
        return (40.0 - 0.2 * z) / (2.0 * z + 40.0) * 720

    counter = VehicleCounter()
    line_a, line_b = row_for(4.0), row_for(20.0)
    # Véhicule à vitesse constante 1 m par image à 10 images/s = 36 km/h
    for frame, z in enumerate(range(2, 23)):
        y = row_for(float(z))  # bas de la boîte = point au sol
        counter.update_tracks(
            [{'id': 7, 'bbox': [300, int(round(y)) - 40, 340, int(round(y))], 'class': 'car', 'conf': 0.9}],
            (0, int(line_a)), (640, int(line_a)), timeline=frame * 0.1, confirm_frames=1,
            speed_line_start=(0, int(line_b)), speed_line_end=(640, int(line_b)),
            speed_distance_m=16.0, ground_distance=ground,
        )
    speed = counter.get_stats()['max_speed_kmh']
    assert abs(speed - 36.0) < 2.5, speed


def test_uncounted_vehicles_are_explained():
    counter = VehicleCounter()
    line_start, line_end = (0, 300), (1000, 300)
    size = (1000, 600)
    paths = {
        1: [(500, y) for y in (250, 230, 210, 190, 170, 150)],   # monte, apparu déjà au-delà
        2: [(x, 450) for x in (60, 40, 20, 5)],                  # sort par le bord gauche
        3: [(500, 500), (500, 498)],                              # trop bref : jamais retenu
        4: [(500, y) for y in (500, 480, 460, 440, 420)],         # disparaît au milieu
        5: [(700, y) for y in (560, 540, 520, 500, 480)],         # encore visible
    }
    for frame in range(6):
        dets = []
        for tid, pts in paths.items():
            if frame < len(pts) or tid == 5:
                x, y = pts[min(frame, len(pts) - 1)]
                if tid == 5:
                    y = 560 - 20 * frame
                dets.append({'id': tid, 'bbox': [x - 10, y - 10, x + 10, y + 10], 'class': 'car', 'conf': 0.9})
        counter.update_tracks(dets, line_start, line_end, timeline=frame * 0.25,
                              confirm_frames=1, frame_size=size)
    reasons = {r['reason']: r['count'] for r in counter.get_stats()['uncounted_reasons']}
    assert reasons == {'already_past': 1, 'still_visible': 3}, reasons  # disparus mais pas encore expirés

    for frame in range(6, 100):  # seul le véhicule 5 reste visible, les autres expirent
        y = max(320, 560 - 20 * frame)
        counter.update_tracks([{'id': 5, 'bbox': [690, y - 10, 710, y + 10], 'class': 'car', 'conf': 0.9}],
                              line_start, line_end, timeline=frame * 0.25, confirm_frames=1, frame_size=size)
    reasons = {r['reason']: r['count'] for r in counter.get_stats()['uncounted_reasons']}
    assert reasons == {'already_past': 1, 'exited_edge': 1, 'lost': 1, 'still_visible': 1}, reasons
    assert counter.get_stats()['total'] == 4


def test_fast_persons_become_two_wheelers_pedestrians_do_not():
    from core.riders import RiderDetector
    riders = RiderDetector()
    bus = {'id': 50, 'bbox': [0, 0, 400, 300], 'class': 'bus', 'conf': 0.9}
    out = []
    for frame in range(12):
        t = frame * 0.1
        dets = [
            bus,
            # piéton : 0,3 hauteur/s
            {'id': 1, 'bbox': [600, 400, 630, int(480 + frame * 0.24)], 'class': 'person', 'conf': 0.8},
            # conducteur + passager sur le même scooter : ~1,5 hauteur/s
            {'id': 2, 'bbox': [900, 300 + frame * 12, 940, 380 + frame * 12], 'class': 'person', 'conf': 0.7},
            {'id': 3, 'bbox': [910, 310 + frame * 12, 950, 390 + frame * 12], 'class': 'person', 'conf': 0.6},
            # passager visible à travers une vitre du bus, qui roule vite
            {'id': 4, 'bbox': [100 + frame * 15, 100, 130 + frame * 15, 160], 'class': 'person', 'conf': 0.5},
        ]
        out = riders.process(dets, t)
    motos = [d for d in out if d['class'] == 'motorcycle']
    assert [d['id'] for d in motos] == [2], motos   # un seul deux-roues, pas de piéton
    assert all(d['class'] != 'person' for d in out)  # les personnes ne sont jamais des véhicules
    assert any(d['id'] == 50 for d in out)


def test_track_fragments_are_merged_and_class_is_stabilized():
    counter = VehicleCounter()
    line = ((0, 100), (1000, 100))

    def frame(dets, t):
        counter.update_tracks(dets, *line, timeline=t, confirm_frames=1,
                              frame_size=(1000, 600), merge_fragments=True)

    def car(tid, y, cls='car', x=500):
        return {'id': tid, 'bbox': [x - 30, y - 20, x + 30, y + 20], 'class': cls, 'conf': 0.8}

    # Véhicule 1 : monte à 100 px/s ; YOLO hésite d'abord sur la catégorie
    for k, cls in enumerate(['truck'] + ['car'] * 7):
        frame([car(1, 500 - 10 * k, cls)], k * 0.1)
    # Perdu 0,5 s, puis réapparaît sous un nouveau numéro à la position prédite
    for k in range(8, 13):
        frame([], k * 0.1)
    for k in range(13, 21):
        frame([car(7, 500 - 10 * k), car(9, 300, x=100)], k * 0.1)  # 9 : un autre véhicule, ailleurs

    stats = counter.get_stats()
    assert stats['total'] == 2, stats['total']                  # 1 (+ fragment 7) et 9
    assert stats['by_class']['car']['total'] == 2
    assert stats['by_class']['truck']['total'] == 0              # « truck » isolé corrigé
    assert counter.track_aliases == {7: 1}


def test_center_landing_exactly_on_line_is_counted():
    counter = VehicleCounter()
    line_start, line_end = (0, 251), (1280, 251)
    for frame, cy in enumerate([275, 267, 259, 251, 243, 235]):
        counter.update_tracks(
            [{'id': 3, 'bbox': [400, cy - 40, 440, cy + 40], 'class': 'car', 'conf': 0.9}],
            line_start, line_end, timeline=frame * 0.04, confirm_frames=1)
    assert counter.get_stats()['count_up'] == 1


def test_counted_vehicles_get_make_and_model():
    import json
    import time
    import types
    import numpy as np
    from core.config import AppConfig
    from core.vehicle_id import VehicleIdentifier

    sent = []

    class FakeMessages:
        def create(self, **kwargs):
            sent.append(kwargs)
            text = json.dumps({'make': 'Toyota', 'model': 'Corolla', 'color': 'blanc',
                               'body_type': 'berline', 'confidence': 'moyenne'})
            return types.SimpleNamespace(stop_reason='end_turn',
                                         content=[types.SimpleNamespace(type='text', text=text)])

    fake_client = types.SimpleNamespace(beta=types.SimpleNamespace(messages=FakeMessages()))
    cfg = AppConfig(vehicle_id_enabled=True)
    config = types.SimpleNamespace(get=lambda: cfg)
    counter = VehicleCounter()
    ident = VehicleIdentifier(config, counter, client_factory=lambda key: fake_client)
    frame = np.zeros((600, 1000, 3), np.uint8)
    line = ((0, 300), (1000, 300))

    # Le véhicule 1 monte et rapetisse : seule sa plus grande image doit partir.
    for k in range(20):
        y, half = 500 - 15 * k, 80 - 3 * k
        dets = [{'id': 1, 'bbox': [500 - half, y - half // 2, 500 + half, y + half // 2], 'class': 'car', 'conf': 0.9},
                {'id': 2, 'bbox': [100, 500, 120, 515], 'class': 'car', 'conf': 0.9}]  # immobile, jamais compté
        counter.update_tracks(dets, *line, timeline=k * 0.2, confirm_frames=1)
        ident.observe(frame, dets, k * 0.2, cfg)
    for k in range(20, 35):  # véhicule sorti du champ : envoi après le délai
        counter.update_tracks([], *line, timeline=k * 0.2, confirm_frames=1)
        ident.observe(frame, [], k * 0.2, cfg)
    for _ in range(1000):
        if any((ev.get('identity') or {}).get('status') == 'done'
               for ev in counter.get_stats()['recent_events']):
            break
        time.sleep(0.02)

    events = {ev['id']: ev for ev in counter.get_stats()['recent_events']}
    assert events[1]['identity']['status'] == 'done', events[1]['identity']
    assert events[1]['identity']['make'] == 'Toyota'
    assert events[2]['identity'] is None                     # non compté : rien envoyé
    assert len(sent) == 1
    assert sent[0]['model'] == 'claude-opus-5-5'
    image = sent[0]['messages'][0]['content'][0]
    assert image['type'] == 'image' and image['source']['media_type'] == 'image/jpeg'
    assert ident.status(cfg)['top_makes'] == [{'make': 'Toyota', 'count': 1}]


def test_vehicle_vanishing_with_known_direction_is_counted():
    counter = VehicleCounter()
    line = ((0, 100), (1000, 100))
    for k in range(10):
        dets = [{'id': 1, 'bbox': [480, 480 - 15 * k, 520, 520 - 15 * k], 'class': 'car', 'conf': 0.9},   # monte
                {'id': 2, 'bbox': [200, 300, 240, 340], 'class': 'car', 'conf': 0.9}]                     # immobile
        counter.update_tracks(dets, *line, timeline=k * 0.2, confirm_frames=1, count_exits=True)
    # Les deux disparaissent sans avoir atteint la ligne
    for k in range(10, 14):  # moins de 2 s : peut encore réapparaître
        counter.update_tracks([], *line, timeline=k * 0.2, confirm_frames=1, count_exits=True)
    assert counter.get_stats()['count_up'] == 0
    for k in range(14, 30):
        counter.update_tracks([], *line, timeline=k * 0.2, confirm_frames=1, count_exits=True)
    stats = counter.get_stats()
    assert stats['count_up'] == 1 and stats['count_down'] == 0
    events = {ev['id']: ev for ev in stats['recent_events']}
    assert events[1]['direction'] == 'up' and events[1]['counted_by'] == 'exit'
    assert events[2]['direction'] is None                       # sans sens établi : pas compté

if __name__ == "__main__":
    print("Exécution des tests de comptage directionnel...")
    test_ccw_and_intersect()
    print("✅ test_ccw_and_intersect validé")
    test_vehicle_counter_direction_up_and_down()
    print("✅ test_vehicle_counter_direction_up_and_down validé (UP, DOWN, Anti-doublon)")
    test_slow_inference_does_not_expire_tracks_and_video_flow_uses_media_time()
    print("✅ test_slow_inference validé (timeline vidéo et débit)")
    test_detection_is_counted_before_direction()
    print("✅ test_detection_before_direction validé")
    test_speed_between_two_calibrated_lines()
    print("✅ test_speed_calibration validé (km/h)")
    test_stable_direction_before_line_survives_immediate_disappearance()
    print("✅ test_direction_before_crossing validé")
    test_pending_statistics_distinguish_before_line_and_uncertain_direction()
    print("✅ test_pending_breakdown validé")
    test_finished_vehicles_leave_memory_but_keep_statistics()
    print("✅ test_memory_bounded validé (agrégats conservés)")
    test_speed_with_perspective_correction()
    print("✅ test_speed_perspective validé (vitesse constante malgré la perspective)")
    test_uncounted_vehicles_are_explained()
    print("✅ test_uncounted_reasons validé (diagnostic des non-comptés)")
    test_fast_persons_become_two_wheelers_pedestrians_do_not()
    print("✅ test_riders validé (deux-roues déduits des personnes rapides)")
    test_track_fragments_are_merged_and_class_is_stabilized()
    print("✅ test_fragments validé (fragments raccordés, catégorie stabilisée)")
    test_center_landing_exactly_on_line_is_counted()
    print("✅ test_ligne validé (centre posé pile sur la ligne compté)")
    test_counted_vehicles_get_make_and_model()
    print("✅ test_marque validé (marque et modèle des véhicules comptés)")
    test_vehicle_vanishing_with_known_direction_is_counted()
    print("✅ test_sortie validé (véhicule disparu dans un sens établi compté)")
    print("🎉 Tous les tests de logique de comptage ont réussi avec succès !")
