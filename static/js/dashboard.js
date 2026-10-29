/**
 * KreyatiSky TrafficVision - Dashboard Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    // Éléments du DOM - Statistiques
    const statCountUp = document.getElementById('statCountUp');
    const statCountDown = document.getElementById('statCountDown');
    const statCountTotal = document.getElementById('statCountTotal');
    const statAwaitingCrossing = document.getElementById('statAwaitingCrossing');
    const statAwaitingDirection = document.getElementById('statAwaitingDirection');
    const statDirectionKnown = document.getElementById('statDirectionKnown');
    const statFlowRate = document.getElementById('statFlowRate');
    const statFps = document.getElementById('statFps');
    const statMaxSpeed = document.getElementById('statMaxSpeed');
    const statAvgSpeed = document.getElementById('statAvgSpeed');
    const statSpeedCount = document.getElementById('statSpeedCount');
    const statSpeedStatus = document.getElementById('statSpeedStatus');
    const pctUp = document.getElementById('pctUp');
    const pctDown = document.getElementById('pctDown');
    const sourceStatusText = document.getElementById('sourceStatusText');
    const statusBadge = document.getElementById('statusBadge');
    const eventsList = document.getElementById('eventsList');

    // Barres de répartition par classe
    const cntClassCar = document.getElementById('cntClassCar');
    const barClassCar = document.getElementById('barClassCar');
    const cntClassTruck = document.getElementById('cntClassTruck');
    const barClassTruck = document.getElementById('barClassTruck');
    const cntClassMotorcycle = document.getElementById('cntClassMotorcycle');
    const barClassMotorcycle = document.getElementById('barClassMotorcycle');
    const cntClassBus = document.getElementById('cntClassBus');
    const barClassBus = document.getElementById('barClassBus');

    // Boutons d'actions
    const btnSnapshot = document.getElementById('btnSnapshot');
    const btnExportCsv = document.getElementById('btnExportCsv');
    const btnReset = document.getElementById('btnReset');
    const btnSettings = document.getElementById('btnSettings');
    const btnCloseSettings = document.getElementById('btnCloseSettings');
    const settingsDrawer = document.getElementById('settingsDrawer');
    const settingsBackdrop = document.getElementById('settingsBackdrop');

    // Contrôles rapides du lecteur vidéo
    const videoContainer = document.getElementById('videoContainer');
    const btnFullscreen = document.getElementById('btnFullscreen');
    const btnToggleBoxes = document.getElementById('btnToggleBoxes');
    const btnToggleTrails = document.getElementById('btnToggleTrails');
    const btnToggleLines = document.getElementById('btnToggleLines');
    const quickSliderLineY = document.getElementById('quickSliderLineY');
    const lblLineY = document.getElementById('lblLineY');

    // Paramètres du Drawer
    const selSourceType = document.getElementById('selSourceType');
    const txtSourcePath = document.getElementById('txtSourcePath');
    const grpSourcePath = document.getElementById('grpSourcePath');
    const grpFileUpload = document.getElementById('grpFileUpload');
    const grpExistingVideos = document.getElementById('grpExistingVideos');
    const selUploadedVideos = document.getElementById('selUploadedVideos');
    const fileUploadInput = document.getElementById('fileUploadInput');
    const btnApplySource = document.getElementById('btnApplySource');
    const chkVehicleId = document.getElementById('chkVehicleId');
    const txtVehicleIdKey = document.getElementById('txtVehicleIdKey');
    const txtVehicleIdRate = document.getElementById('txtVehicleIdRate');
    const lblVehicleIdKey = document.getElementById('lblVehicleIdKey');
    const lblVehicleIdStatus = document.getElementById('lblVehicleIdStatus');

    const sliderConf = document.getElementById('sliderConf');
    const lblConf = document.getElementById('lblConf');
    const chkCar = document.getElementById('chkCar');
    const chkTruck = document.getElementById('chkTruck');
    const chkMotorcycle = document.getElementById('chkMotorcycle');
    const chkBus = document.getElementById('chkBus');

    const chkShowBoxes = document.getElementById('chkShowBoxes');
    const chkShowTrails = document.getElementById('chkShowTrails');
    const chkShowLabels = document.getElementById('chkShowLabels');
    const chkShowLine = document.getElementById('chkShowLine');
    const txtSpeedDistance = document.getElementById('txtSpeedDistance');
    const sliderSpeedLineY = document.getElementById('sliderSpeedLineY');
    const lblSpeedLineY = document.getElementById('lblSpeedLineY');
    const chkShowSpeedLine = document.getElementById('chkShowSpeedLine');
    const chkAutoPerformance = document.getElementById('chkAutoPerformance');
    const txtTargetFps = document.getElementById('txtTargetFps');
    const grpTargetFps = document.getElementById('grpTargetFps');
    const btnSaveSettings = document.getElementById('btnSaveSettings');

    let currentConfig = {};
    let debounceTimer = null;


    // Éléments du lecteur de fichier vidéo
    const videoFileControls = document.getElementById('videoFileControls');
    const btnVideoPlayPause = document.getElementById('btnVideoPlayPause');
    const iconPlayPause = document.getElementById('iconPlayPause');
    const btnVideoReplay = document.getElementById('btnVideoReplay');
    const txtVideoStatus = document.getElementById('txtVideoStatus');
    const lblVideoCurrentTime = document.getElementById('lblVideoCurrentTime');
    const lblVideoTotalTime = document.getElementById('lblVideoTotalTime');
    const lblVideoPercent = document.getElementById('lblVideoPercent');
    const barVideoProgress = document.getElementById('barVideoProgress');

    // Modale Bilan de fin de vidéo
    const modalVideoComplete = document.getElementById('modalVideoComplete');
    const modalVideoBackdrop = document.getElementById('modalVideoBackdrop');
    const modalTotalCount = document.getElementById('modalTotalCount');
    const modalUpCount = document.getElementById('modalUpCount');
    const modalDownCount = document.getElementById('modalDownCount');
    const modalCarCount = document.getElementById('modalCarCount');
    const modalTruckCount = document.getElementById('modalTruckCount');
    const modalMotorcycleCount = document.getElementById('modalMotorcycleCount');
    const modalBusCount = document.getElementById('modalBusCount');
    const btnModalReplay = document.getElementById('btnModalReplay');
    const btnModalExport = document.getElementById('btnModalExport');
    const btnModalClose = document.getElementById('btnModalClose');

    let hasShownCompletionModal = false;


    // --- 1. Rafraîchissement des Statistiques en temps réel ---
    async function fetchStats() {
        try {
            const res = await fetch('/api/stats');
            if (!res.ok) return;
            const data = await res.json();

            // Compteurs principaux
            const up = data.count_up || 0;
            const down = data.count_down || 0;
            const total = data.total || 0;

            statCountUp.textContent = up;
            statCountDown.textContent = down;
            statCountTotal.textContent = total;
            statAwaitingCrossing.textContent = data.count_awaiting_crossing || 0;
            statAwaitingDirection.textContent = data.count_awaiting_direction || 0;
            statDirectionKnown.textContent = data.direction_known_count || 0;
            statFlowRate.textContent = data.flow_rate_vpm || '0.0';
            const reducedWidth = data.inference_width && data.inference_width < data.configured_width;
            statFps.textContent = `${data.auto_performance ? 'AUTO · ' : ''}${data.fps || '0.0'} FPS`
                + (data.capacity_fps ? ` · max ${data.capacity_fps}` : '')
                + (reducedWidth ? ` · ${data.inference_width}px` : '');
            const deviceNames = { cpu: 'processeur', mps: 'puce graphique Apple' };
            const deviceName = data.compute_device?.startsWith('cuda')
                ? 'carte graphique NVIDIA' : (deviceNames[data.compute_device] || data.compute_device || '');
            statFps.title = data.capacity_fps
                ? `Débit obtenu : ${data.fps} FPS. Capacité maximale mesurée de cet appareil : ${data.capacity_fps} FPS (${data.processing_ms} ms de calcul par image, sur ${deviceName}).`
                  + (reducedWidth ? ` Analyse réduite automatiquement à ${data.inference_width} px (au lieu de ${data.configured_width} px) pour garder un suivi fiable en direct.` : '')
                : '';
            const measuredSpeeds = data.speed_vehicles_count || 0;
            statMaxSpeed.textContent = measuredSpeeds ? Number(data.max_speed_kmh).toFixed(1) : '—';
            statAvgSpeed.textContent = measuredSpeeds ? `${Number(data.avg_speed_kmh).toFixed(1)} km/h` : '— km/h';
            statSpeedCount.textContent = `${measuredSpeeds} mesurée${measuredSpeeds > 1 ? 's' : ''}`;
            const calibration = `${Number(data.speed_calibration_distance_m || 0).toFixed(1)} m`
                + (data.speed_perspective ? ' · perspective corrigée' : '');
            statSpeedStatus.textContent = measuredSpeeds
                ? `Mesure active · ${calibration}`
                : `En attente · zone de ${calibration}`;

            // Pourcentages
            if (total > 0) {
                pctUp.textContent = `${Math.round((up / total) * 100)}%`;
                pctDown.textContent = `${Math.round((down / total) * 100)}%`;
            } else {
                pctUp.textContent = '0%';
                pctDown.textContent = '0%';
            }

            // Statut Source
            if (data.source_status) {
                sourceStatusText.textContent = data.source_status;
            }

            // Gestion de la progression du fichier vidéo
            const vp = data.video_progress;
            if (vp && vp.is_video_file) {
                videoFileControls.classList.remove('hidden');
                lblVideoCurrentTime.textContent = vp.current_time || '00:00';
                lblVideoTotalTime.textContent = vp.total_time || '00:00';
                lblVideoPercent.textContent = `(${vp.percent}%)`;
                barVideoProgress.style.width = `${vp.percent}%`;

                if (vp.is_finished) {
                    txtVideoStatus.textContent = "✅ Analyse terminée !";
                    txtVideoStatus.className = "text-xs font-semibold text-emerald-400";
                    if (iconPlayPause) {
                        iconPlayPause.setAttribute('data-lucide', 'rotate-ccw');
                    }
                    // Affichage automatique de la modale de bilan complet
                    if (!hasShownCompletionModal) {
                        showCompletionModal(data);
                        hasShownCompletionModal = true;
                    }
                } else if (vp.is_paused) {
                    txtVideoStatus.textContent = "⏸️ En pause";
                    txtVideoStatus.className = "text-xs font-semibold text-amber-400";
                    if (iconPlayPause) {
                        iconPlayPause.setAttribute('data-lucide', 'play');
                    }
                } else {
                    txtVideoStatus.textContent = "▶️ Lecture & Analyse...";
                    txtVideoStatus.className = "text-xs font-semibold text-sky-400";
                    if (iconPlayPause) {
                        iconPlayPause.setAttribute('data-lucide', 'pause');
                    }
                }
            } else {
                videoFileControls.classList.add('hidden');
                hasShownCompletionModal = false;
            }

            // Répartition par classe
            const bc = data.by_class || {};
            updateClassBar('car', bc.car, barClassCar, cntClassCar, total);
            updateClassBar('truck', bc.truck, barClassTruck, cntClassTruck, total);
            updateClassBar('motorcycle', bc.motorcycle, barClassMotorcycle, cntClassMotorcycle, total);
            updateClassBar('bus', bc.bus, barClassBus, cntClassBus, total);

            // Journal des événements récents
            renderEvents(data.recent_events || []);
            renderVehicleIdStatus(data.vehicle_id);
        } catch (e) {
            console.error("Erreur stats:", e);
        }
    }

    function showCompletionModal(data) {
        const bc = data.by_class || {};
        modalTotalCount.textContent = data.total || 0;
        modalUpCount.textContent = data.count_up || 0;
        modalDownCount.textContent = data.count_down || 0;

        const classLine = (c) => {
            const counted = (c?.up || 0) + (c?.down || 0);
            return `${c?.total || 0} détectés · ${counted} comptés (▲${c?.up || 0} | ▼${c?.down || 0})`;
        };
        modalCarCount.textContent = classLine(bc.car);
        modalTruckCount.textContent = classLine(bc.truck);
        modalMotorcycleCount.textContent = classLine(bc.motorcycle);
        modalBusCount.textContent = classLine(bc.bus);

        const reasons = data.uncounted_reasons || [];
        const uncountedBox = document.getElementById('modalUncountedBox');
        uncountedBox.classList.toggle('hidden', reasons.length === 0);
        document.getElementById('modalUncountedTotal').textContent =
            reasons.reduce((sum, r) => sum + r.count, 0);
        document.getElementById('modalUncountedList').innerHTML = reasons.map(r => `
            <div class="flex justify-between gap-3 text-slate-300">
                <span>${escapeHtml(r.label)}</span>
                <span class="font-bold text-slate-200">${escapeHtml(r.count)}</span>
            </div>`).join('');

        const makes = data.vehicle_id?.top_makes || [];
        document.getElementById('modalMakesBox').classList.toggle('hidden', makes.length === 0);
        document.getElementById('modalMakesList').innerHTML = makes.map(m => `
            <div class="flex justify-between gap-3 text-slate-300">
                <span>${escapeHtml(m.make)}</span>
                <span class="font-bold text-slate-200">${escapeHtml(m.count)}</span>
            </div>`).join('');

        modalVideoComplete.classList.remove('hidden');
        modalVideoComplete.classList.add('flex');
        if (window.lucide) {
            lucide.createIcons();
        }
    }

    function hideCompletionModal() {
        if (modalVideoComplete) {
            modalVideoComplete.classList.add('hidden');
            modalVideoComplete.classList.remove('flex');
        }
    }

    btnModalClose.addEventListener('click', hideCompletionModal);

    if (modalVideoBackdrop) {
        modalVideoBackdrop.addEventListener('click', hideCompletionModal);
    }



    btnModalReplay.addEventListener('click', async () => {
        hideCompletionModal();
        hasShownCompletionModal = false;
        await fetch('/api/video/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'replay' })
        });
        fetchStats();
    });


    btnModalExport.addEventListener('click', () => {
        window.location.href = '/api/export/csv';
    });

    // Boutons de contrôle de lecture vidéo
    btnVideoPlayPause.addEventListener('click', async () => {
        const res = await fetch('/api/video/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'pause' })
        });
        const d = await res.json();
        if (window.lucide) lucide.createIcons();
    });

    btnVideoReplay.addEventListener('click', async () => {
        hasShownCompletionModal = false;
        await fetch('/api/video/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'replay' })
        });
        fetchStats();
    });


    function updateClassBar(key, stat, barEl, txtEl, total) {
        if (!stat) return;
        const tot = stat.total || 0;
        const up = stat.up || 0;
        const down = stat.down || 0;
        txtEl.textContent = `${tot} (▲${up} | ▼${down})`;
        const pct = total > 0 ? Math.round((tot / total) * 100) : 0;
        barEl.style.width = `${pct}%`;
    }

    function renderEvents(events) {
        if (events.length === 0) {
            eventsList.innerHTML = `<div class="text-center py-10 text-xs text-slate-500">En attente de détection de véhicules...</div>`;
            return;
        }

        const iconsMap = {
            car: 'car',
            truck: 'truck',
            motorcycle: 'bike',
            bus: 'bus'
        };

        const namesMap = {
            car: 'Voiture',
            truck: 'Camion',
            motorcycle: 'Moto',
            bus: 'Bus'
        };

        eventsList.innerHTML = events.slice(0, 15).map(ev => {
            const displayedDirection = ev.direction || ev.motion_direction;
            const isUp = displayedDirection === 'up';
            const isDown = displayedDirection === 'down';
            const dirColor = isUp
                ? 'text-emerald-400 bg-emerald-950/80 border-emerald-800'
                : isDown
                    ? 'text-orange-400 bg-orange-950/80 border-orange-800'
                    : 'text-sky-300 bg-sky-950/80 border-sky-800';
            // Seuls les franchissements de ligne entrent dans les cartes Monte/Descend.
            // Un sens simplement estimé reste visiblement distinct : contour pointillé, « non compté ».
            const estimated = !ev.direction && !!ev.motion_direction;
            const badgeStyle = estimated ? 'border-dashed opacity-70 !bg-transparent' : '';
            const byExit = ev.counted_by === 'exit';
            const dirLabel = isUp
                ? estimated ? '↑ monte · non compté' : byExit ? '▲ MONTE · compté (sorti)' : '▲ MONTE · compté'
                : isDown
                    ? estimated ? '↓ descend · non compté' : byExit ? '▼ DESCEND · compté (sorti)' : '▼ DESCEND · compté'
                    : ev.status === 'direction_pending' ? '◌ ligne franchie · sens incertain' : '● avant la ligne';
            const dirTitle = estimated
                ? 'Sens estimé d’après le mouvement, pas encore compté : il le sera dans ce sens s’il disparaît sans franchir la ligne.'
                : byExit
                    ? 'A disparu sans franchir la ligne, dans un sens bien établi : compté dans ce sens.'
                : ev.direction
                    ? 'A franchi la ligne jaune : compté dans les cartes Monte/Descend.'
                    : ev.status === 'direction_pending'
                        ? 'A franchi la ligne, mais le mouvement n’est pas encore assez net pour décider du sens.'
                        : 'Détecté, pas encore arrivé à la ligne jaune.';
            const iconName = iconsMap[ev.class] || 'car';
            const frName = namesMap[ev.class] || ev.class;
            const identityLabel = renderIdentity(ev.identity);
            const speedLabel = ev.max_speed_kmh != null
                ? `<span class="text-pink-300 font-semibold">${Number(ev.max_speed_kmh).toFixed(1)} km/h</span>`
                : `<span class="text-slate-500">vitesse en mesure</span>`;

            return `
                <div class="flex items-center justify-between p-2 rounded-xl bg-slate-900/70 border border-slate-800/80 text-xs hover:border-slate-700 transition">
                    <div class="flex items-center space-x-2">
                        <span class="w-6 h-6 rounded-lg bg-slate-800 flex items-center justify-center text-slate-300">
                            <i data-lucide="${iconName}" class="w-3.5 h-3.5"></i>
                        </span>
                        <div>
                            <span class="font-bold text-slate-200">${frName}</span>
                            <span class="text-[10px] text-slate-500 ml-1">#${ev.id}</span>
                            ${identityLabel}
                        </div>
                    </div>
                    <div class="flex items-center space-x-2">
                        <span class="px-2 py-0.5 rounded-full text-[10px] font-bold border ${dirColor} ${badgeStyle}" title="${dirTitle}">
                            ${dirLabel}
                        </span>
                        <span class="text-[11px] font-mono text-slate-400">${ev.time_str}</span>
                        ${speedLabel}
                    </div>
                </div>
            `;
        }).join('');

        if (window.lucide) {
            lucide.createIcons();
        }
    }

    // Intervalle de mise à jour toutes les 600ms
    setInterval(fetchStats, 600);
    fetchStats();

    // --- 2. Actions des boutons principaux ---
    btnSnapshot.addEventListener('click', () => {
        window.open('/api/snapshot', '_blank');
    });

    btnExportCsv.addEventListener('click', () => {
        window.location.href = '/api/export/csv';
    });

    btnReset.addEventListener('click', async () => {
        if (confirm('Voulez-vous vraiment réinitialiser tous les compteurs ?')) {
            hideCompletionModal();
            hasShownCompletionModal = false;
            await fetch('/api/reset', { method: 'POST' });
            fetchStats();
        }
    });



    btnFullscreen.addEventListener('click', () => {
        if (!document.fullscreenElement) {
            videoContainer.requestFullscreen().catch(err => {
                console.error("Erreur plein écran:", err);
            });
        } else {
            document.exitFullscreen();
        }
    });

    // Curseur rapide pour la ligne de comptage
    quickSliderLineY.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value) / 100.0;
        lblLineY.textContent = `${e.target.value}%`;

        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(async () => {
            await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    line_start_y: val,
                    line_end_y: val
                })
            });
        }, 150);
    });

    // Bascules rapides Boîtes & Trajectoires
    btnToggleBoxes.addEventListener('click', async () => {
        const current = currentConfig.show_boxes !== false;
        const res = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ show_boxes: !current })
        });
        const data = await res.json();
        currentConfig = data.config;
        btnToggleBoxes.classList.toggle('bg-sky-600', !current);
    });

    btnToggleTrails.addEventListener('click', async () => {
        const current = currentConfig.show_trails !== false;
        const res = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ show_trails: !current })
        });
        const data = await res.json();
        currentConfig = data.config;
        btnToggleTrails.classList.toggle('bg-sky-600', !current);
    });

    // Masque seulement le dessin des lignes : elles restent actives pour le comptage et la vitesse.
    btnToggleLines.addEventListener('click', async () => {
        const visible = !(currentConfig.show_line !== false || currentConfig.show_speed_line !== false);
        const res = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ show_line: visible, show_speed_line: visible })
        });
        const data = await res.json();
        currentConfig = data.config;
        chkShowLine.checked = visible;
        chkShowSpeedLine.checked = visible;
        updateLinesButton();
    });

    function updateLinesButton() {
        const visible = currentConfig.show_line !== false || currentConfig.show_speed_line !== false;
        btnToggleLines.classList.toggle('bg-sky-600', visible);
        btnToggleLines.textContent = visible ? 'Lignes' : 'Lignes (masquées)';
    }

    // --- 3. Drawer des Paramètres ---

    function openSettings() {
        hideCompletionModal();
        loadSettings();
        settingsDrawer.classList.remove('hidden');
    }



    function closeSettings() {
        settingsDrawer.classList.add('hidden');
    }

    btnSettings.addEventListener('click', openSettings);
    btnCloseSettings.addEventListener('click', closeSettings);
    settingsBackdrop.addEventListener('click', closeSettings);

    // Les noms de fichiers viennent du disque du serveur : ne jamais les injecter tels quels.
    function renderIdentity(identity) {
        if (!identity) return '';
        if (identity.status === 'pending') {
            return '<div class="text-[10px] text-amber-300/70">identification…</div>';
        }
        if (identity.status === 'done') {
            const known = [identity.make, identity.model].filter(v => v && v.toLowerCase() !== 'inconnu');
            const name = known.length ? known.join(' ') : 'modèle non reconnu';
            const details = [identity.color, `fiabilité ${identity.confidence}`].filter(Boolean).join(' · ');
            return `<div class="text-[10px] text-amber-300" title="${escapeHtml(identity.body_type || '')}">${escapeHtml(name)} <span class="text-slate-500">· ${escapeHtml(details)}</span></div>`;
        }
        const reasons = {
            too_small: 'trop petit pour être identifié',
            quota: 'non identifié (limite par minute)',
            error: identity.error || 'échec de l’identification',
        };
        return `<div class="text-[10px] text-slate-500">${escapeHtml(reasons[identity.status] || '')}</div>`;
    }

    function renderVehicleIdStatus(status) {
        if (!status || !lblVehicleIdStatus) return;
        if (!status.enabled) {
            lblVehicleIdStatus.textContent = '';
            return;
        }
        const parts = [`${status.identified} identifié(s)`];
        if (status.queued) parts.push(`${status.queued} en attente`);
        if (status.skipped) parts.push(`${status.skipped} ignoré(s)`);
        if (status.failed) parts.push(`${status.failed} échec(s)`);
        lblVehicleIdStatus.textContent = parts.join(' · ') + (status.last_error ? ` — ${status.last_error}` : '');
        lblVehicleIdStatus.classList.toggle('text-red-400', !!status.last_error);
    }

    function escapeHtml(value) {
        return String(value).replace(/[&<>"']/g, ch => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        })[ch]);
    }

    async function loadUploadedVideos(selectedPath) {
        try {
            const res = await fetch('/api/uploaded_videos');
            if (!res.ok) return;
            const data = await res.json();
            const videos = data.videos || [];

            if (videos.length > 0 && selUploadedVideos) {
                grpExistingVideos.classList.remove('hidden');
                selUploadedVideos.innerHTML = `<option value="">-- Choisir une vidéo existante (${videos.length} disponible${videos.length > 1 ? 's' : ''}) --</option>` +
                    videos.map(v => `<option value="${escapeHtml(v.path)}" ${v.path === selectedPath ? 'selected' : ''}>🎬 ${escapeHtml(v.name)} (${escapeHtml(v.size_mb)} MB)</option>`).join('');
            } else if (grpExistingVideos) {
                grpExistingVideos.classList.add('hidden');
            }
        } catch (e) {
            console.error("Erreur chargement vidéos:", e);
        }
    }

    if (selUploadedVideos) {
        selUploadedVideos.addEventListener('change', (e) => {
            if (e.target.value) {
                txtSourcePath.value = e.target.value;
            }
        });
    }

    async function loadSettings() {
        try {
            const res = await fetch('/api/settings');
            const cfg = await res.json();
            currentConfig = cfg;

            selSourceType.value = cfg.source_type || 'demo';
            txtSourcePath.value = cfg.source_path || '0';
            updateSourceInputs(cfg.source_type);

            await loadUploadedVideos(cfg.source_path);

            sliderConf.value = Math.round(cfg.confidence * 100);
            lblConf.textContent = `${sliderConf.value}%`;

            const classes = cfg.selected_classes || ['car', 'motorcycle', 'bus', 'truck'];
            chkCar.checked = classes.includes('car');
            chkTruck.checked = classes.includes('truck');
            chkMotorcycle.checked = classes.includes('motorcycle');
            chkBus.checked = classes.includes('bus');

            chkShowBoxes.checked = cfg.show_boxes !== false;
            chkShowTrails.checked = cfg.show_trails !== false;
            chkShowLabels.checked = cfg.show_labels !== false;
            chkShowLine.checked = cfg.show_line !== false;
            txtSpeedDistance.value = cfg.speed_distance_m || 10;
            sliderSpeedLineY.value = Math.round((cfg.speed_line_y ?? 0.55) * 100);
            lblSpeedLineY.textContent = `${sliderSpeedLineY.value}%`;
            chkShowSpeedLine.checked = cfg.show_speed_line !== false;
            chkAutoPerformance.checked = cfg.auto_performance !== false;
            txtTargetFps.value = cfg.target_fps || 12;
            grpTargetFps.classList.toggle('hidden', chkAutoPerformance.checked);
            chkVehicleId.checked = cfg.vehicle_id_enabled === true;
            txtVehicleIdKey.value = '';
            txtVehicleIdRate.value = cfg.vehicle_id_max_per_minute || 20;
            lblVehicleIdKey.textContent = cfg.vehicle_id_key_set
                ? 'Clé enregistrée. Laisser vide pour la garder.'
                : 'Aucune clé enregistrée : ANTHROPIC_API_KEY sera utilisée si elle existe.';

            updateLinesButton();
            quickSliderLineY.value = Math.round(cfg.line_start_y * 100);
            lblLineY.textContent = `${quickSliderLineY.value}%`;
        } catch (e) {
            console.error("Erreur chargement paramètres:", e);
        }
    }

    function updateSourceInputs(type) {
        if (type === 'demo') {
            grpSourcePath.classList.add('hidden');
            grpFileUpload.classList.add('hidden');
        } else if (type === 'file') {
            grpSourcePath.classList.remove('hidden');
            grpFileUpload.classList.remove('hidden');
            txtSourcePath.placeholder = "/chemin/vers/video.mp4";
            loadUploadedVideos(txtSourcePath.value);
        } else if (type === 'webcam') {
            grpSourcePath.classList.remove('hidden');
            grpFileUpload.classList.add('hidden');
            txtSourcePath.placeholder = "0 pour webcam intégrée, 1 pour USB";
        } else if (type === 'csi') {
            grpSourcePath.classList.remove('hidden');
            grpFileUpload.classList.add('hidden');
            txtSourcePath.placeholder = "0 pour la première caméra CSI, 1 pour la seconde";
        } else if (type === 'rtsp') {
            grpSourcePath.classList.remove('hidden');
            grpFileUpload.classList.add('hidden');
            txtSourcePath.placeholder = "rtsp://admin:pass@192.168.1.100:554/stream";
        }
    }

    selSourceType.addEventListener('change', (e) => {
        updateSourceInputs(e.target.value);
    });

    sliderConf.addEventListener('input', (e) => {
        lblConf.textContent = `${e.target.value}%`;
    });

    sliderSpeedLineY.addEventListener('input', (e) => {
        lblSpeedLineY.textContent = `${e.target.value}%`;
    });

    chkAutoPerformance.addEventListener('change', () => {
        grpTargetFps.classList.toggle('hidden', chkAutoPerformance.checked);
    });

    // Appliquer changement de source
    btnApplySource.addEventListener('click', async () => {
        const type = selSourceType.value;
        let path = txtSourcePath.value.trim();

        // 1. Si téléversement d'un nouveau fichier vidéo
        if (type === 'file' && fileUploadInput.files.length > 0) {
            const formData = new FormData();
            formData.append('file', fileUploadInput.files[0]);
            btnApplySource.textContent = "Téléversement et démarrage...";
            
            try {
                const upRes = await fetch('/api/upload_video', {
                    method: 'POST',
                    body: formData
                });
                const upData = await upRes.json();
                btnApplySource.textContent = "Lancer cette Source Vidéo";

                if (upData.status === 'success') {
                    txtSourcePath.value = upData.file_path;
                    sourceStatusText.textContent = upData.source_status || `Fichier: ${upData.filename}`;
                    hasShownCompletionModal = false;
                    closeSettings();
                    fetchStats();
                    return;
                } else {
                    alert(`Erreur lors du téléversement : ${upData.message}`);
                    return;
                }
            } catch (err) {
                btnApplySource.textContent = "Lancer cette Source Vidéo";
                alert("Erreur réseau lors du téléversement de la vidéo.");
                return;
            }
        }

        // 2. Si sélection d'un fichier existant ou webcam / RTSP / Démo
        if (type === 'file' && !path && selUploadedVideos && selUploadedVideos.value) {
            path = selUploadedVideos.value;
            txtSourcePath.value = path;
        }

        btnApplySource.textContent = "Application...";
        const res = await fetch('/api/source', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source_type: type,
                source_path: path || '0'
            })
        });
        const data = await res.json();
        btnApplySource.textContent = "Lancer cette Source Vidéo";
        
        if (data.status === 'success' || data.status === 'pending') {
            sourceStatusText.textContent = data.source_status;
            hasShownCompletionModal = false;
            closeSettings();
            fetchStats();
            if (data.status === 'pending') {
                alert(`Source enregistrée, connexion en cours : ${data.message}`);
            }
        } else {
            alert(`Erreur : ${data.message}`);
        }
    });

    // Enregistrer les paramètres complets
    btnSaveSettings.addEventListener('click', async () => {
        const selectedClasses = [];
        if (chkCar.checked) selectedClasses.push('car');
        if (chkTruck.checked) selectedClasses.push('truck');
        if (chkMotorcycle.checked) selectedClasses.push('motorcycle');
        if (chkBus.checked) selectedClasses.push('bus');

        const payload = {
            source_type: selSourceType.value,
            source_path: txtSourcePath.value.trim() || '0',
            confidence: parseFloat(sliderConf.value) / 100.0,
            selected_classes: selectedClasses,
            show_boxes: chkShowBoxes.checked,
            show_trails: chkShowTrails.checked,
            show_labels: chkShowLabels.checked,
            show_line: chkShowLine.checked,
            speed_distance_m: parseFloat(txtSpeedDistance.value),
            speed_line_y: parseFloat(sliderSpeedLineY.value) / 100.0,
            show_speed_line: chkShowSpeedLine.checked,
            auto_performance: chkAutoPerformance.checked,
            target_fps: parseInt(txtTargetFps.value, 10) || 12,
            vehicle_id_enabled: chkVehicleId.checked,
            vehicle_id_api_key: txtVehicleIdKey.value.trim(),
            vehicle_id_max_per_minute: parseInt(txtVehicleIdRate.value, 10) || 20
        };

        const res = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (data.status === 'success') {
            currentConfig = data.config;
            closeSettings();
            fetchStats();
        } else {
            alert(`Erreur : ${data.message}`);
        }
    });

    // Initialisation
    loadSettings();
});
