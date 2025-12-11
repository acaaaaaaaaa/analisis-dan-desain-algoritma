"""
===================================================================================
APLIKASI PENJADWALAN MAHASISWA OTOMATIS
===================================================================================
Aplikasi ini membantu mahasiswa mengatur jadwal kegiatan dengan berbagai algoritma
optimasi. Mendukung CRUD, visualisasi kalender, dan export jadwal.

Author: Assistant
Date: 2025
===================================================================================
"""

# ===================================================================================
# IMPORT LIBRARIES
# ===================================================================================
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import io
from typing import List, Dict, Tuple
import calendar
import itertools

# ===================================================================================
# KONFIGURASI STREAMLIT PAGE
# ===================================================================================
st.set_page_config(
    page_title="📅 Penjadwalan Mahasiswa",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================================================================
# CSS CUSTOM STYLING - TEMA ORANYE + COLOR PSYCHOLOGY
# ===================================================================================
# Oranye: energi, kreativitas, semangat
# Biru: produktivitas, fokus
# Hijau: kesehatan, keseimbangan
# Merah: urgent, deadline
st.markdown("""
<style>
    .main {
        background-color: #FFF5EE;
    }
    .stButton>button {
        background-color: #FF8C00;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF7F00;
    }
    .header-style {
        font-size: 36px;
        font-weight: bold;
        color: #FF6347;
        text-align: center;
        padding: 20px;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .deadline-urgent {
        background-color: #FFE4E1;
        border-left: 5px solid #FF4500;
    }
    .task-completed {
        background-color: #F0FFF0;
        border-left: 5px solid #32CD32;
    }
</style>
""", unsafe_allow_html=True)

# ===================================================================================
# CLASS: TASK (DATA MODEL UNTUK TUGAS/KEGIATAN)
# ===================================================================================
class Task:
    """
    Class untuk merepresentasikan satu tugas/kegiatan mahasiswa.
    
    Attributes:
        id: Unique identifier
        name: Nama kegiatan
        category: Kategori (Kuliah, Tugas, Ujian, Pribadi, dll)
        start_time: Waktu mulai
        duration: Durasi dalam jam
        deadline: Batas waktu
        priority: Prioritas (1-5, 5 = paling penting)
        is_fixed: Apakah waktu sudah tetap (misal: jadwal kuliah)
    """
    # Perbaikan: Mengubah _init_ menjadi __init__
    def __init__(self, id, name, category, start_time, duration, deadline, priority=3, is_fixed=False):
        self.id = id
        self.name = name
        self.category = category
        self.start_time = start_time
        self.duration = duration  # dalam jam
        self.deadline = deadline
        self.priority = priority
        self.is_fixed = is_fixed
        self.end_time = start_time + timedelta(hours=duration)
    
    def to_dict(self):
        """Convert task object ke dictionary untuk export"""
        return {
            'ID': self.id,
            'Nama': self.name,
            'Kategori': self.category,
            'Waktu Mulai': self.start_time.strftime('%Y-%m-%d %H:%M'),
            'Durasi (jam)': self.duration,
            # Deadline harus di-format sebagai datetime.date untuk menghindari error
            'Deadline': self.deadline.strftime('%Y-%m-%d'), 
            'Prioritas': self.priority,
            'Fixed': self.is_fixed
        }

# ===================================================================================
# FUNGSI: INISIALISASI SESSION STATE
# ===================================================================================
def init_session_state():
    """
    Inisialisasi session state untuk menyimpan data persistent
    Session state memungkinkan data tetap tersimpan saat user berinteraksi
    """
    if 'tasks' not in st.session_state:
        st.session_state.tasks = []
    if 'task_counter' not in st.session_state:
        st.session_state.task_counter = 1
    if 'scheduled_tasks' not in st.session_state:
        st.session_state.scheduled_tasks = []
    if 'algorithm_used' not in st.session_state:
        st.session_state.algorithm_used = None

# ===================================================================================
# ALGORITMA 1: BRUTE FORCE SCHEDULING
# ===================================================================================
def brute_force_scheduling(tasks: List[Task], start_date: datetime, end_date: datetime) -> List[Task]:
    """
    Algoritma Brute Force: mencoba semua kemungkinan kombinasi jadwal.
    
    Kompleksitas: O(n!) - sangat lambat untuk n > 10
    """
    if len(tasks) == 0:
        return []
    
    # Pisahkan tugas fixed dan flexible
    fixed_tasks = [t for t in tasks if t.is_fixed]
    flexible_tasks = [t for t in tasks if not t.is_fixed]
    
    if len(flexible_tasks) > 10:
        st.warning("⚠ Brute Force tidak efisien untuk > 10 tugas flexible. Menggunakan sample 10 tugas pertama.")
        flexible_tasks = flexible_tasks[:10]
    
    best_schedule = None
    best_score = float('inf')
    
    # Coba semua permutasi tugas flexible
    for perm in itertools.permutations(flexible_tasks):
        current_time = start_date
        scheduled = fixed_tasks.copy()
        penalty = 0
        
        for task in perm:
            # Skip waktu yang sudah dipakai tugas fixed
            while any(current_time < t.end_time and current_time + timedelta(hours=task.duration) > t.start_time 
                     for t in fixed_tasks):
                current_time += timedelta(hours=1)
            
            # Jadwalkan tugas
            # Hati-hati: Di sini kita membuat objek Task baru,
            # pastikan properti end_time ikut terupdate secara otomatis (sudah di-handle oleh __init__)
            new_task = Task(
                task.id, task.name, task.category,
                current_time, task.duration, task.deadline,
                task.priority, task.is_fixed
            )
            
            # Hitung penalty jika melewati deadline
            if new_task.end_time > task.deadline:
                penalty += (new_task.end_time - task.deadline).days * task.priority * 10
            
            scheduled.append(new_task)
            current_time = new_task.end_time
        
        # Update best schedule jika lebih baik
        if penalty < best_score:
            best_score = penalty
            best_schedule = scheduled
    
    return best_schedule if best_schedule else fixed_tasks

# ===================================================================================
# ALGORITMA 2: GREEDY EARLIEST DEADLINE FIRST (EDF)
# ===================================================================================
def greedy_edf_scheduling(tasks: List[Task], start_date: datetime, end_date: datetime) -> List[Task]:
    """
    Algoritma Greedy - Earliest Deadline First (EDF)
    
    Kompleksitas: O(n log n)
    """
    if len(tasks) == 0:
        return []
    
    # Pisahkan tugas fixed dan flexible
    fixed_tasks = [t for t in tasks if t.is_fixed]
    flexible_tasks = [t for t in tasks if not t.is_fixed]
    
    # Sort flexible tasks berdasarkan deadline, lalu prioritas
    flexible_tasks.sort(key=lambda x: (x.deadline, -x.priority))
    
    scheduled = fixed_tasks.copy()
    current_time = start_date
    
    for task in flexible_tasks:
        # Cari waktu kosong berikutnya
        while True:
            # Cek apakah waktu ini bentrok dengan tugas yang sudah dijadwalkan
            conflict = False
            proposed_end = current_time + timedelta(hours=task.duration)
            
            # Periksa konflik dengan SEMUA tugas yang sudah dijadwalkan (termasuk fixed)
            for scheduled_task in scheduled:
                if (current_time < scheduled_task.end_time and 
                    proposed_end > scheduled_task.start_time):
                    conflict = True
                    # Majukan current_time ke waktu selesai task yang konflik
                    current_time = scheduled_task.end_time
                    break
            
            if not conflict:
                break
            
            # Tambahkan jeda waktu minimal 1 menit
            current_time += timedelta(minutes=1) 
    
        # Pastikan tidak melewati batas akhir penjadwalan
        if current_time + timedelta(hours=task.duration) > end_date:
            continue # Lewati tugas jika tidak bisa dijadwalkan dalam batas waktu
            
        # Jadwalkan tugas di waktu yang ditemukan
        new_task = Task(
            task.id, task.name, task.category,
            current_time, task.duration, task.deadline,
            task.priority, task.is_fixed
        )
        scheduled.append(new_task)
        current_time = new_task.end_time
    
    return scheduled

# ===================================================================================
# ALGORITMA 3: DYNAMIC PROGRAMMING
# ===================================================================================
def dynamic_programming_scheduling(tasks: List[Task], start_date: datetime, end_date: datetime) -> List[Task]:
    """
    Algoritma Dynamic Programming untuk Weighted Interval Scheduling (Perlu asumsi start_time tugas flexible)
    
    Kompleksitas: O(n²)
    """
    if len(tasks) == 0:
        return []
    
    # Pisahkan fixed dan flexible
    fixed_tasks = [t for t in tasks if t.is_fixed]
    flexible_tasks = [t for t in tasks if not t.is_fixed]
    
    if len(flexible_tasks) == 0:
        return fixed_tasks
    
    # *Modifikasi*: Untuk DP interval scheduling, tugas harus memiliki waktu mulai/selesai yang jelas
    # Kita menggunakan Greedy EDF untuk memberikan *start time* awal yang realistis pada tugas flexible.
    temp_scheduled = greedy_edf_scheduling(flexible_tasks, start_date, end_date)
    
    # Sort berdasarkan end time
    temp_scheduled.sort(key=lambda x: x.end_time)
    
    n = len(temp_scheduled)
    
    # Hitung weight (value) setiap tugas: prioritas * (1 + bonus_karena_waktu_luang_sebelum_deadline)
    weights = []
    for task in temp_scheduled:
        time_to_deadline = (task.deadline - task.end_time).total_seconds() / 3600
        # Bobot lebih tinggi jika tugas diselesaikan jauh dari deadline
        weight = task.priority * (1 + max(0, time_to_deadline / (24*7))) 
        weights.append(weight)
    
    # Cari tugas kompatibel terakhir untuk setiap tugas
    def find_latest_compatible(index):
        # Cari tugas j < index yang end_time-nya <= start_time tugas index
        for j in range(index - 1, -1, -1):
            if temp_scheduled[j].end_time <= temp_scheduled[index].start_time:
                return j
        return -1
    
    # DP array: dp[i] = maximum value achievable dengan tugas 0..i
    dp = [0] * n
    dp[0] = weights[0]
    
    for i in range(1, n):
        # Pilihan 1: Ambil tugas i
        include = weights[i]
        latest_compatible = find_latest_compatible(i)
        if latest_compatible != -1:
            include += dp[latest_compatible]
        
        # Pilihan 2: Skip tugas i
        exclude = dp[i - 1]
        
        dp[i] = max(include, exclude)
    
    # Backtrack untuk menemukan tugas yang dipilih
    selected = []
    i = n - 1
    while i >= 0:
        if i == 0:
            if dp[i] > 0: # Hanya ambil jika bobot positif
                selected.append(temp_scheduled[i])
            break
        
        # Hitung ulang nilai include dan exclude untuk pengambilan keputusan
        latest_compatible = find_latest_compatible(i)
        include_val = weights[i] + (dp[latest_compatible] if latest_compatible != -1 else 0)
        exclude_val = dp[i - 1]
        
        if include_val >= exclude_val:
            selected.append(temp_scheduled[i])
            i = latest_compatible if latest_compatible != -1 else -1 # Pindah ke tugas kompatibel terakhir
        else:
            i -= 1
    
    # Reverse karena backtrack dari belakang
    selected.reverse()
    
    # Gabungkan dengan fixed tasks
    all_tasks = fixed_tasks + selected
    
    # Final check and re-sort
    all_tasks.sort(key=lambda x: x.start_time)
    
    return all_tasks

# ===================================================================================
# ALGORITMA 4: GENETIC ALGORITHM (DEFAULT - PALING OPTIMAL)
# ===================================================================================
def genetic_algorithm_scheduling(tasks: List[Task], start_date: datetime, end_date: datetime, 
                                 population_size=50, generations=100) -> List[Task]:
    """
    Algoritma Genetika untuk Optimasi Penjadwalan
    
    Kompleksitas: O(g * p * n)
    """
    if len(tasks) == 0:
        return []
    
    fixed_tasks = [t for t in tasks if t.is_fixed]
    flexible_tasks = [t for t in tasks if not t.is_fixed]
    
    if len(flexible_tasks) == 0:
        return fixed_tasks
    
    # Fungsi fitness: semakin kecil (cost) semakin baik
    def calculate_fitness(schedule):
        fitness = 0
        
        # Penalty 1: Konflik waktu (harus diminimalisir)
        for i, task1 in enumerate(schedule):
            for task2 in schedule[i+1:]:
                if (task1.start_time < task2.end_time and 
                    task1.end_time > task2.start_time):
                    overlap = min(task1.end_time, task2.end_time) - max(task1.start_time, task2.start_time)
                    fitness += overlap.total_seconds() / 3600 * 100  # penalty per jam overlap
        
        # Penalty 2: Melewati deadline
        for task in schedule:
            if task.end_time > task.deadline:
                days_late = (task.end_time - task.deadline).days
                fitness += days_late * task.priority * 50
        
        # Bonus (dikurangi dari fitness) untuk menyelesaikan task high priority lebih awal
        for task in schedule:
            if task.end_time <= task.deadline:
                days_early = (task.deadline - task.end_time).days
                fitness -= days_early * task.priority * 2
        
        return fitness
    
    # Inisialisasi populasi
    def create_individual():
        schedule = fixed_tasks.copy()
        shuffled = flexible_tasks.copy()
        random.shuffle(shuffled)
        
        for task in shuffled:
            # Random start time dalam range kerja (misalnya 8 pagi hingga 8 malam)
            days_range = (end_date - start_date).days
            if days_range <= 0: days_range = 1
            
            random_days = random.randint(0, max(0, days_range - 1))
            random_hours = random.randint(8, 20 - int(task.duration)) # Jam kerja 8-20
            
            start = start_date + timedelta(days=random_days, hours=random_hours)
            
            # Cek konflik (sederhana) untuk inisialisasi awal
            new_task = Task(
                task.id, task.name, task.category,
                start, task.duration, task.deadline,
                task.priority, task.is_fixed
            )
            schedule.append(new_task)
        
        return schedule
    
    population = [create_individual() for _ in range(population_size)]
    
    # Evolusi
    for generation in range(generations):
        # Evaluasi fitness
        fitness_scores = [(schedule, calculate_fitness(schedule)) for schedule in population]
        fitness_scores.sort(key=lambda x: x[1])
        
        # Elitism: keep top 10%
        elite_count = max(2, population_size // 10)
        new_population = [schedule for schedule, _ in fitness_scores[:elite_count]]
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 5
            parent1 = min(random.sample(fitness_scores, tournament_size), key=lambda x: x[1])[0]
            parent2 = min(random.sample(fitness_scores, tournament_size), key=lambda x: x[1])[0]
            
            # Crossover (Order Crossover)
            crossover_point = len(flexible_tasks) // 2
            
            # 1. Ambil ID dari parent1
            parent1_ids = {t.id for t in parent1 if not t.is_fixed}
            parent2_flexible = [t for t in parent2 if not t.is_fixed]

            # 2. Ciptakan jadwal anak (child)
            child_flexible_map = {}
            
            # Crossover: Ambil waktu/posisi dari Parent1 untuk setengah tugas pertama (berdasarkan urutan ID di P1)
            p1_flex = [t for t in parent1 if not t.is_fixed]
            for i in range(crossover_point):
                task = p1_flex[i]
                child_flexible_map[task.id] = task

            # Crossover: Ambil waktu/posisi dari Parent2 untuk tugas sisanya
            # Gunakan tugas dari Parent2 yang belum ada di child
            for task in parent2_flexible:
                if task.id not in child_flexible_map:
                    child_flexible_map[task.id] = task

            child_flexible = list(child_flexible_map.values())
            
            # Mutasi: 10% chance untuk random reschedule satu tugas
            if random.random() < 0.1 and len(child_flexible) > 0:
                mutate_idx = random.randint(0, len(child_flexible) - 1)
                task = child_flexible[mutate_idx]
                
                days_range = (end_date - start_date).days
                random_days = random.randint(0, max(0, days_range - 1))
                random_hours = random.randint(8, 20)
                
                new_start = start_date + timedelta(days=random_days, hours=random_hours)
                child_flexible[mutate_idx] = Task(
                    task.id, task.name, task.category,
                    new_start, task.duration, task.deadline,
                    task.priority, task.is_fixed
                )
            
            child = fixed_tasks + child_flexible
            new_population.append(child)
        
        population = new_population
    
    # Return jadwal terbaik
    best_schedule = min(population, key=calculate_fitness)
    best_schedule.sort(key=lambda x: x.start_time)
    
    return best_schedule

# ===================================================================================
# FUNGSI: DETEKSI KONFLIK JADWAL
# ===================================================================================
def detect_conflicts(tasks: List[Task]) -> List[Tuple[Task, Task]]:
    """
    Deteksi konflik waktu antar tugas
    
    Returns:
        List of tuples berisi pasangan tugas yang konflik
    """
    conflicts = []
    # Urutkan tugas berdasarkan waktu mulai untuk efisiensi
    tasks.sort(key=lambda x: x.start_time) 
    
    for i, task1 in enumerate(tasks):
        for task2 in tasks[i+1:]:
            if (task1.start_time < task2.end_time and 
                task1.end_time > task2.start_time):
                conflicts.append((task1, task2))
    return conflicts

# ===================================================================================
# FUNGSI: HITUNG STATISTIK JADWAL
# ===================================================================================
def calculate_statistics(tasks: List[Task], start_date: datetime, end_date: datetime) -> Dict:
    """
    Hitung statistik dari jadwal yang dibuat
    
    Returns:
        Dictionary berisi berbagai metrik jadwal
    """
    if not tasks:
        return {
            'total_tasks': 0,
            'total_hours': 0,
            'free_hours': 0,
            'conflicts': 0,
            'late_tasks': 0,
            'on_time_tasks': 0
        }
    
    total_hours = sum(t.duration for t in tasks)
    total_available_hours = (end_date - start_date).days * 24
    free_hours = max(0, total_available_hours - total_hours)
    
    conflicts = len(detect_conflicts(tasks))
    late_tasks = sum(1 for t in tasks if t.end_time > t.deadline)
    on_time_tasks = len(tasks) - late_tasks
    
    return {
        'total_tasks': len(tasks),
        'total_hours': total_hours,
        'free_hours': free_hours,
        'conflicts': conflicts,
        'late_tasks': late_tasks,
        'on_time_tasks': on_time_tasks
    }

# ===================================================================================
# FUNGSI: EXPORT KE CSV
# ===================================================================================
def export_to_csv(tasks: List[Task]) -> str:
    """
    Export jadwal ke format CSV
    
    Returns:
        String CSV content
    """
    df = pd.DataFrame([t.to_dict() for t in tasks])
    return df.to_csv(index=False)

# ===================================================================================
# FUNGSI: CEK DEADLINE MENDEKAT
# ===================================================================================
def check_upcoming_deadlines(tasks: List[Task], days_threshold=3) -> List[Task]:
    """
    Cek tugas dengan deadline mendekat dalam N hari
    """
    now = datetime.now()
    upcoming = []
    
    for task in tasks:
        # Cek hanya tanggal (abaikan waktu) untuk deadline
        days_until_deadline = (task.deadline.date() - now.date()).days
        if 0 <= days_until_deadline <= days_threshold:
            upcoming.append(task)
    
    return upcoming

# ===================================================================================
# FUNGSI: DAPATKAN WARNA BERDASARKAN KATEGORI
# ===================================================================================
def get_category_color(category: str) -> str:
    """
    Return warna hex berdasarkan kategori (Color Psychology)
    """
    color_map = {
        'Kuliah': '#4169E1',       # Royal Blue
        'Tugas': '#FF8C00',        # Dark Orange
        'Ujian': '#DC143C',        # Crimson
        'Pribadi': '#32CD32',      # Lime Green
        'Organisasi': '#9370DB',   # Medium Purple
        'Olahraga': '#20B2AA',     # Light Sea Green
        'Istirahat': '#87CEEB',    # Sky Blue
        'Makan': '#FFD700',        # Gold
    }
    return color_map.get(category, '#808080')  # Default: Gray

# ===================================================================================
# FUNGSI: DAPATKAN EMOJI BERDASARKAN KATEGORI
# ===================================================================================
def get_category_emoji(category: str) -> str:
    """Return emoji yang sesuai dengan kategori"""
    emoji_map = {
        'Kuliah': '📚',
        'Tugas': '✍️',
        'Ujian': '📝',
        'Pribadi': '🏠',
        'Organisasi': '👥',
        'Olahraga': '⚽',
        'Istirahat': '😴',
        'Makan': '🍽️',
    }
    return emoji_map.get(category, '📌')

# ===================================================================================
# MAIN APPLICATION
# ===================================================================================
def main():
    """Fungsi utama aplikasi"""
    
    # Inisialisasi session state
    init_session_state()
    
    # Header
    st.markdown('<div class="header-style">📅 Sistem Penjadwalan Mahasiswa Otomatis</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar untuk navigasi
    st.sidebar.title("🎯 Menu Navigasi")
    menu = st.sidebar.radio(
        "Pilih Menu:",
        ["📝 Input & Manage Tugas", "🤖 Generate Jadwal", "📊 Lihat Jadwal", "📥 Export & Notifikasi"]
    )
    
    # ===================================================================================
    # MENU 1: INPUT & MANAGE TUGAS
    # ===================================================================================
    if menu == "📝 Input & Manage Tugas":
        st.header("📝 Manajemen Tugas & Kegiatan")
        
        tab1, tab2, tab3 = st.tabs(["➕ Tambah Tugas", "📋 Lihat & Edit", "📤 Import CSV"])
        
        # TAB: Tambah Tugas Manual
        with tab1:
            st.subheader("Tambah Tugas Baru")
            
            col1, col2 = st.columns(2)
            
            with col1:
                task_name = st.text_input("Nama Kegiatan", placeholder="contoh: Kuliah Algoritma")
                category = st.selectbox(
                    "Kategori",
                    ["Kuliah", "Tugas", "Ujian", "Pribadi", "Organisasi", "Olahraga", "Istirahat", "Makan"]
                )
                priority = st.slider("Prioritas (1=rendah, 5=tinggi)", 1, 5, 3)
                is_fixed = st.checkbox("Waktu Fixed (tidak bisa diubah oleh algoritma)", value=False)
            
            with col2:
                start_date = st.date_input("Tanggal Mulai", datetime.now().date()) # Ambil hanya tanggal
                start_time = st.time_input("Jam Mulai", datetime.now().time())
                duration = st.number_input("Durasi (jam)", min_value=0.5, max_value=24.0, value=2.0, step=0.5)
                deadline_date = st.date_input("Deadline", datetime.now().date() + timedelta(days=7)) # Ambil hanya tanggal
            
            if st.button("➕ Tambah Tugas", type="primary"):
                # Pastikan input nama tidak kosong
                if not task_name:
                    st.error("Nama kegiatan tidak boleh kosong!")
                else:
                    start_datetime = datetime.combine(start_date, start_time)
                    # Deadline menggunakan waktu maksimal hari itu
                    deadline_datetime = datetime.combine(deadline_date, datetime.max.time())
                    
                    new_task = Task(
                        id=st.session_state.task_counter,
                        name=task_name,
                        category=category,
                        start_time=start_datetime,
                        duration=duration,
                        deadline=deadline_datetime,
                        priority=priority,
                        is_fixed=is_fixed
                    )
                    
                    st.session_state.tasks.append(new_task)
                    st.session_state.task_counter += 1
                    st.success(f"✅ Tugas '{task_name}' berhasil ditambahkan!")
        
        # TAB: Lihat & Edit
        with tab2:
            st.subheader("Daftar Tugas")
            
            if len(st.session_state.tasks) == 0:
                st.info("📭 Belum ada tugas. Silakan tambah tugas terlebih dahulu.")
            else:
                # Tampilkan dalam tabel
                tasks_data = [t.to_dict() for t in st.session_state.tasks]
                df = pd.DataFrame(tasks_data)
                st.dataframe(df, use_container_width=True, height=300)
                
                col1, col2 = st.columns(2)
                
                task_options = range(len(st.session_state.tasks))
                
                # EDIT TUGAS
                with col1:
                    st.subheader("✏️ Edit Tugas")
                    task_to_edit_index = st.selectbox(
                        "Pilih tugas yang akan diedit:",
                        options=task_options,
                        format_func=lambda x: f"{st.session_state.tasks[x].name} - {st.session_state.tasks[x].category}",
                        key="edit_select"
                    )
                    
                    if task_to_edit_index is not None:
                        selected_task = st.session_state.tasks[task_to_edit_index]
                        
                        with st.form(key="edit_form"):
                            edit_name = st.text_input("Nama Kegiatan", value=selected_task.name)
                            category_options = ["Kuliah", "Tugas", "Ujian", "Pribadi", "Organisasi", "Olahraga", "Istirahat", "Makan"]
                            edit_category = st.selectbox(
                                "Kategori",
                                category_options,
                                index=category_options.index(selected_task.category) if selected_task.category in category_options else 0
                            )
                            
                            edit_col1, edit_col2 = st.columns(2)
                            with edit_col1:
                                edit_start_date = st.date_input("Tanggal Mulai", value=selected_task.start_time.date())
                                edit_start_time = st.time_input("Jam Mulai", value=selected_task.start_time.time())
                            with edit_col2:
                                edit_duration = st.number_input("Durasi (jam)", min_value=0.5, max_value=24.0, value=float(selected_task.duration), step=0.5)
                                edit_deadline = st.date_input("Deadline", value=selected_task.deadline.date())
                            
                            edit_priority = st.slider("Prioritas", 1, 5, selected_task.priority)
                            edit_is_fixed = st.checkbox("Waktu Fixed", value=selected_task.is_fixed)
                            
                            submit_edit = st.form_submit_button("💾 Simpan Perubahan", type="primary")
                            
                            if submit_edit:
                                # Update task
                                st.session_state.tasks[task_to_edit_index].name = edit_name
                                st.session_state.tasks[task_to_edit_index].category = edit_category
                                st.session_state.tasks[task_to_edit_index].start_time = datetime.combine(edit_start_date, edit_start_time)
                                st.session_state.tasks[task_to_edit_index].duration = edit_duration
                                st.session_state.tasks[task_to_edit_index].deadline = datetime.combine(edit_deadline, datetime.max.time())
                                st.session_state.tasks[task_to_edit_index].priority = edit_priority
                                st.session_state.tasks[task_to_edit_index].is_fixed = edit_is_fixed
                                st.session_state.tasks[task_to_edit_index].end_time = st.session_state.tasks[task_to_edit_index].start_time + timedelta(hours=edit_duration)
                                
                                st.success(f"✅ Tugas '{edit_name}' berhasil diupdate!")
                                # st.rerun() # Tidak perlu rerun karena data sudah diupdate di session state
                
                # HAPUS TUGAS
                with col2:
                    st.subheader("🗑️ Hapus Tugas")
                    task_to_delete_index = st.selectbox(
                        "Pilih tugas yang akan dihapus:",
                        options=task_options,
                        format_func=lambda x: f"{st.session_state.tasks[x].name} - {st.session_state.tasks[x].category}",
                        key="delete_select"
                    )
                    
                    if task_to_delete_index is not None:
                        task_name_to_delete = st.session_state.tasks[task_to_delete_index].name
                        
                        if st.button(f"🗑️ Konfirmasi Hapus '{task_name_to_delete}'"):
                            del st.session_state.tasks[task_to_delete_index]
                            st.success(f"🗑️ Tugas '{task_name_to_delete}' berhasil dihapus.")
                            st.rerun()
                            
        # TAB: Import CSV (Placeholder)
        with tab3:
            st.info("Fitur Import CSV akan ditambahkan di versi mendatang.")
            # ... (Logika Import CSV)
            
    # ===================================================================================
    # MENU 2: GENERATE JADWAL (ALGORITMA)
    # ===================================================================================
    if menu == "🤖 Generate Jadwal":
        st.header("⚙️ Optimasi Penjadwalan")
        
        if not st.session_state.tasks:
            st.warning("Tambahkan tugas terlebih dahulu di menu 'Input & Manage Tugas'.")
            return
            
        st.info(f"Total {len(st.session_state.tasks)} tugas siap dijadwalkan.")
            
        # Pilihan Algoritma
        algorithm_choice = st.selectbox(
            "Pilih Algoritma Penjadwalan:",
            [
                "Genetic Algorithm (GA) - Optimal untuk Kompleksitas",
                "Greedy Earliest Deadline First (EDF) - Cepat & Prioritas Deadline",
                "Dynamic Programming (DP) - Optimal untuk Nilai Tugas",
                "Brute Force - Optimal Tapi Sangat Lambat (Max 10 Tugas Fleksibel)"
            ],
            key="algo_select"
        )

        # Parameter Penjadwalan
        st.subheader("Rentang Waktu Penjadwalan")
        col1, col2 = st.columns(2)
        with col1:
            schedule_start_date = st.date_input("Mulai Dari Tanggal", datetime.now().date())
        with col2:
            schedule_end_date = st.date_input("Hingga Tanggal", datetime.now().date() + timedelta(days=7))
            
        start_dt = datetime.combine(schedule_start_date, datetime.min.time())
        end_dt = datetime.combine(schedule_end_date, datetime.max.time())
        
        st.markdown("---")
        
        if st.button("🚀 Generate Jadwal", type="primary"):
            st.session_state.scheduled_tasks = [] # Reset jadwal sebelumnya
            st.session_state.algorithm_used = algorithm_choice
            
            with st.spinner(f"Sedang menjalankan {algorithm_choice}..."):
                if "Brute Force" in algorithm_choice:
                    scheduled = brute_force_scheduling(st.session_state.tasks, start_dt, end_dt)
                elif "Greedy" in algorithm_choice:
                    scheduled = greedy_edf_scheduling(st.session_state.tasks, start_dt, end_dt)
                elif "Dynamic Programming" in algorithm_choice:
                    scheduled = dynamic_programming_scheduling(st.session_state.tasks, start_dt, end_dt)
                elif "Genetic" in algorithm_choice:
                    scheduled = genetic_algorithm_scheduling(st.session_state.tasks, start_dt, end_dt)
                else:
                    scheduled = []

            st.session_state.scheduled_tasks = scheduled
            
            if scheduled:
                stats = calculate_statistics(scheduled, start_dt, end_dt)
                st.success(f"🎉 Penjadwalan Selesai menggunakan **{algorithm_choice}**!")
                st.metric("Tugas Selesai Tepat Waktu", f"{stats['on_time_tasks']}/{stats['total_tasks']}", delta=f"-{stats['late_tasks']} terlambat")
                st.metric("Total Konflik Waktu", stats['conflicts'], delta_color="inverse")
            else:
                st.warning("Jadwal tidak dapat dibuat. Coba sesuaikan rentang waktu atau input tugas.")
                
    # ===================================================================================
    # MENU 3: LIHAT JADWAL
    # ===================================================================================
    if menu == "📊 Lihat Jadwal":
        st.header("🗓️ Visualisasi Kalender dan Detail Jadwal")

        if not st.session_state.scheduled_tasks:
            st.info("Silakan *Generate Jadwal* terlebih dahulu di menu sebelumnya.")
            return

        scheduled_tasks = st.session_state.scheduled_tasks
        algorithm = st.session_state.algorithm_used
        
        st.subheader(f"Hasil Optimasi ({algorithm})")
        stats = calculate_statistics(scheduled_tasks, scheduled_tasks[0].start_time, scheduled_tasks[-1].end_time)
        
        col_st1, col_st2, col_st3, col_st4 = st.columns(4)
        col_st1.metric("Total Tugas Dijadwalkan", stats['total_tasks'])
        col_st2.metric("Jam Kerja Terpakai", f"{stats['total_hours']:.1f} jam")
        col_st3.metric("Konflik", stats['conflicts'], delta_color="inverse")
        col_st4.metric("Terlambat Deadline", stats['late_tasks'], delta_color="inverse")
        
        st.markdown("---")
        
        # Tampilkan dalam bentuk Kalender Harian (Gantt Chart Sederhana)
        st.subheader("Visualisasi Jadwal Harian")
        
        # Ambil rentang tanggal dari jadwal yang ada
        min_date = min(t.start_time for t in scheduled_tasks).date()
        max_date = max(t.end_time for t in scheduled_tasks).date()
        date_range = [min_date + timedelta(days=x) for x in range((max_date - min_date).days + 1)]
        
        selected_date = st.date_input("Pilih Tanggal", min_value=min_date, max_value=max_date, value=min_date)
        
        # Filter tugas untuk tanggal yang dipilih
        daily_schedule = [t for t in scheduled_tasks if t.start_time.date() == selected_date]
        daily_schedule.sort(key=lambda x: x.start_time)
        
        if daily_schedule:
            st.markdown(f"### Jadwal **{selected_date.strftime('%A, %d %B %Y')}**")
            
            # Buat representasi visual jam 8 pagi - 8 malam
            for hour in range(8, 21):
                st.markdown(f"**{hour:02}:00**")
                
                # Cek task yang dimulai atau berlangsung pada jam ini
                for task in daily_schedule:
                    start_hour = task.start_time.hour + task.start_time.minute / 60
                    end_hour = task.end_time.hour + task.end_time.minute / 60
                    
                    if start_hour <= hour < end_hour:
                        
                        # Hitung durasi span (untuk visualisasi)
                        # Hitung berapa menit task ini berlangsung di jam ini
                        start_of_current_hour = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=hour)
                        end_of_current_hour = start_of_current_hour + timedelta(hours=1)
                        
                        overlap_start = max(start_of_current_hour, task.start_time)
                        overlap_end = min(end_of_current_hour, task.end_time)
                        
                        if overlap_end > overlap_start:
                            overlap_minutes = (overlap_end - overlap_start).total_seconds() / 60
                            # Proporsionalitas: 1 jam = 100%
                            height_style = f"height: {overlap_minutes * 2}px; line-height: {overlap_minutes * 2}px; overflow: hidden;"
                            
                            color = get_category_color(task.category)
                            emoji = get_category_emoji(task.category)
                            
                            st.markdown(f"""
                            <div style="
                                background-color: {color};
                                color: white;
                                padding: 5px;
                                border-radius: 5px;
                                margin-left: 10px;
                                margin-top: -10px;
                                margin-bottom: 5px;
                                font-size: 14px;
                                {height_style}
                            ">
                                {emoji} {task.name} ({task.start_time.strftime('%H:%M')} - {task.end_time.strftime('%H:%M')})
                            </div>
                            """, unsafe_allow_html=True)
                st.markdown('<div style="border-left: 2px dashed #ccc; height: 10px; margin-left: 8px;"></div>', unsafe_allow_html=True)
        else:
            st.info(f"Tidak ada kegiatan terjadwal pada tanggal {selected_date.strftime('%d %B %Y')}.")
            
    # ===================================================================================
    # MENU 4: EXPORT & NOTIFIKASI
    # ===================================================================================
    if menu == "📥 Export & Notifikasi":
        st.header("📥 Export & Monitoring")

        # NOTIFIKASI DEADLINE
        st.subheader("🔔 Notifikasi Deadline Mendekat")
        
        upcoming_deadlines = check_upcoming_deadlines(st.session_state.tasks, days_threshold=7)
        
        if upcoming_deadlines:
            st.warning(f"🚨 Terdapat **{len(upcoming_deadlines)}** tugas dengan deadline dalam 7 hari ke depan!")
            for task in upcoming_deadlines:
                days_left = (task.deadline.date() - datetime.now().date()).days
                st.markdown(f"""
                <div class="card deadline-urgent">
                    **{task.name}** ({task.category})
                    <br>
                    Deadline: **{task.deadline.strftime('%d %B')}** (Sisa {days_left} hari)
                    <br>
                    Durasi: {task.duration} jam. Prioritas: {task.priority}/5
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ Tidak ada deadline mendesak dalam 7 hari ke depan.")

        st.markdown("---")
        
        # EXPORT JADWAL
        st.subheader("📤 Export Jadwal")
        
        if st.session_state.scheduled_tasks:
            csv_data = export_to_csv(st.session_state.scheduled_tasks)
            st.download_button(
                label="⬇️ Download Jadwal Terakhir (CSV)",
                data=csv_data,
                file_name=f"Jadwal_Mahasiswa_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                type="secondary"
            )
            st.success("Jadwal Anda siap untuk di-export ke CSV.")
        else:
            st.info("Silakan *Generate Jadwal* terlebih dahulu untuk mengaktifkan fitur export.")


# ===================================================================================
# EKSEKUSI UTAMA
# ===================================================================================
# PERBAIKAN KRUSIAL: Mengganti _name_ == "_main_" menjadi __name__ == "__main__"
# (Tambahkan dua garis bawah pada variabel __name__ dan __main__)
if __name__ == "__main__":
    main()
