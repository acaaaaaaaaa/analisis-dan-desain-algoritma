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
    
    Cara kerja:
    1. Generate semua permutasi urutan tugas
    2. Untuk setiap permutasi, jadwalkan secara sekuensial
    3. Pilih jadwal dengan total waktu tunggu terkecil
    
    Kompleksitas: O(n!) - sangat lambat untuk n > 10
    Cocok untuk: dataset kecil (< 10 tugas)
    
    Args:
        tasks: List of Task objects
        start_date: Tanggal mulai penjadwalan
        end_date: Tanggal akhir penjadwalan
    
    Returns:
        List of scheduled Task objects
    """
    if len(tasks) == 0:
        return []
    
    # Pisahkan tugas fixed dan flexible
    fixed_tasks = [t for t in tasks if t.is_fixed]
    flexible_tasks = [t for t in tasks if not t.is_fixed]
    
    if len(flexible_tasks) > 10:
        st.warning("⚠️ Brute Force tidak efisien untuk > 10 tugas flexible. Menggunakan sample.")
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
    
    Cara kerja:
    1. Urutkan tugas berdasarkan deadline (paling dekat dulu)
    2. Jadwalkan secara sekuensial dari waktu paling awal tersedia
    3. Skip waktu yang sudah dipakai tugas fixed
    
    Kompleksitas: O(n log n) untuk sorting + O(n) untuk scheduling = O(n log n)
    Cocok untuk: dataset sedang hingga besar
    Kelebihan: Cepat, intuitif, hasil cukup optimal
    
    Args:
        tasks: List of Task objects
        start_date: Tanggal mulai penjadwalan
        end_date: Tanggal akhir penjadwalan
    
    Returns:
        List of scheduled Task objects
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
            
            for scheduled_task in scheduled:
                if (current_time < scheduled_task.end_time and 
                    proposed_end > scheduled_task.start_time):
                    conflict = True
                    current_time = scheduled_task.end_time
                    break
            
            if not conflict:
                break
        
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
    Algoritma Dynamic Programming untuk Weighted Interval Scheduling
    
    Cara kerja:
    1. Urutkan tugas berdasarkan end time
    2. Untuk setiap tugas, hitung nilai optimal jika tugas diambil atau tidak
    3. Backtrack untuk menemukan kombinasi tugas optimal
    
    Kompleksitas: O(n²) untuk mencari kompatibilitas + O(n) untuk DP = O(n²)
    Cocok untuk: optimasi maksimum value dengan constraint waktu
    Kelebihan: Hasil optimal secara matematis untuk weighted scheduling
    
    Args:
        tasks: List of Task objects
        start_date: Tanggal mulai penjadwalan
        end_date: Tanggal akhir penjadwalan
    
    Returns:
        List of scheduled Task objects
    """
    if len(tasks) == 0:
        return []
    
    # Pisahkan fixed dan flexible
    fixed_tasks = [t for t in tasks if t.is_fixed]
    flexible_tasks = [t for t in tasks if not t.is_fixed]
    
    if len(flexible_tasks) == 0:
        return fixed_tasks
    
    # Assign start time sementara untuk sorting
    current_time = start_date
    for task in flexible_tasks:
        task.start_time = current_time
        task.end_time = current_time + timedelta(hours=task.duration)
        current_time = task.end_time
    
    # Sort berdasarkan end time
    flexible_tasks.sort(key=lambda x: x.end_time)
    n = len(flexible_tasks)
    
    # Hitung weight (value) setiap tugas: prioritas * (1 - keterlambatan)
    weights = []
    for task in flexible_tasks:
        delay_factor = max(0, 1 - (task.end_time - task.deadline).days / 7)
        weight = task.priority * delay_factor
        weights.append(weight)
    
    # Cari tugas kompatibel terakhir untuk setiap tugas
    def find_latest_compatible(index):
        for j in range(index - 1, -1, -1):
            if flexible_tasks[j].end_time <= flexible_tasks[index].start_time:
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
            if dp[i] > 0:
                selected.append(flexible_tasks[i])
            break
        
        # Jika nilai dengan include > exclude, berarti tugas i dipilih
        latest_compatible = find_latest_compatible(i)
        include = weights[i] + (dp[latest_compatible] if latest_compatible != -1 else 0)
        exclude = dp[i - 1]
        
        if include > exclude:
            selected.append(flexible_tasks[i])
            i = latest_compatible
        else:
            i -= 1
    
    # Reverse karena backtrack dari belakang
    selected.reverse()
    
    # Gabungkan dengan fixed tasks dan re-schedule untuk menghindari konflik
    all_tasks = fixed_tasks + selected
    all_tasks.sort(key=lambda x: x.start_time)
    
    return all_tasks

# ===================================================================================
# ALGORITMA 4: GENETIC ALGORITHM (DEFAULT - PALING OPTIMAL)
# ===================================================================================
def genetic_algorithm_scheduling(tasks: List[Task], start_date: datetime, end_date: datetime, 
                                 population_size=50, generations=100) -> List[Task]:
    """
    Algoritma Genetika untuk Optimasi Penjadwalan
    
    Cara kerja:
    1. Inisialisasi populasi jadwal random
    2. Evaluasi fitness setiap jadwal (minimize konflik, deadline miss, maximize priority)
    3. Seleksi jadwal terbaik (tournament selection)
    4. Crossover (combine 2 jadwal parent)
    5. Mutasi (random change untuk eksplorasi)
    6. Repeat untuk beberapa generasi
    
    Kompleksitas: O(g * p * n) dimana g=generasi, p=populasi, n=tugas
    Cocok untuk: dataset besar, optimasi multi-objektif
    Kelebihan: Hasil sangat optimal, dapat handle constraint kompleks
    
    Args:
        tasks: List of Task objects
        start_date: Tanggal mulai penjadwalan
        end_date: Tanggal akhir penjadwalan
        population_size: Ukuran populasi per generasi
        generations: Jumlah iterasi evolusi
    
    Returns:
        List of scheduled Task objects
    """
    if len(tasks) == 0:
        return []
    
    fixed_tasks = [t for t in tasks if t.is_fixed]
    flexible_tasks = [t for t in tasks if not t.is_fixed]
    
    if len(flexible_tasks) == 0:
        return fixed_tasks
    
    # Fungsi fitness: semakin kecil semakin baik
    def calculate_fitness(schedule):
        fitness = 0
        
        # Penalty untuk konflik waktu
        for i, task1 in enumerate(schedule):
            for task2 in schedule[i+1:]:
                if (task1.start_time < task2.end_time and 
                    task1.end_time > task2.start_time):
                    overlap = min(task1.end_time, task2.end_time) - max(task1.start_time, task2.start_time)
                    fitness += overlap.total_seconds() / 3600 * 100  # penalty per jam overlap
        
        # Penalty untuk melewati deadline
        for task in schedule:
            if task.end_time > task.deadline:
                days_late = (task.end_time - task.deadline).days
                fitness += days_late * task.priority * 50
        
        # Bonus untuk menyelesaikan task high priority lebih awal
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
            # Random start time dalam range
            days_range = (end_date - start_date).days
            random_days = random.randint(0, max(0, days_range - 1))
            random_hours = random.randint(8, 20)  # Jam kerja 8-20
            
            start = start_date + timedelta(days=random_days, hours=random_hours)
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
            
            # Crossover
            crossover_point = len(flexible_tasks) // 2
            child_flexible = []
            
            # Ambil setengah dari parent1
            parent1_flexible = [t for t in parent1 if not t.is_fixed][:crossover_point]
            child_flexible.extend(parent1_flexible)
            
            # Ambil sisanya dari parent2 (yang belum ada di child)
            parent2_flexible = [t for t in parent2 if not t.is_fixed]
            used_ids = {t.id for t in child_flexible}
            for t in parent2_flexible:
                if t.id not in used_ids:
                    child_flexible.append(t)
            
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
    
    Args:
        tasks: List of tasks
        days_threshold: Berapa hari ke depan yang dianggap "mendekat"
    
    Returns:
        List of tasks dengan deadline mendekat
    """
    now = datetime.now()
    upcoming = []
    
    for task in tasks:
        days_until_deadline = (task.deadline - now).days
        if 0 <= days_until_deadline <= days_threshold:
            upcoming.append(task)
    
    return upcoming

# ===================================================================================
# FUNGSI: DAPATKAN WARNA BERDASARKAN KATEGORI
# ===================================================================================
def get_category_color(category: str) -> str:
    """
    Return warna hex berdasarkan kategori (Color Psychology)
    
    Kuliah: Biru (fokus, pembelajaran)
    Tugas: Oranye (energi, kreativitas)
    Ujian: Merah (urgent, penting)
    Pribadi: Hijau (keseimbangan, kesehatan)
    Organisasi: Ungu (kolaborasi)
    Lainnya: Abu-abu
    """
    color_map = {
        'Kuliah': '#4169E1',      # Royal Blue
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
                start_date = st.date_input("Tanggal Mulai", datetime.now())
                start_time = st.time_input("Jam Mulai", datetime.now().time())
                duration = st.number_input("Durasi (jam)", min_value=0.5, max_value=24.0, value=2.0, step=0.5)
                deadline_date = st.date_input("Deadline", datetime.now() + timedelta(days=7))
            
            if st.button("➕ Tambah Tugas", type="primary"):
                start_datetime = datetime.combine(start_date, start_time)
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
                st.dataframe(df, use_container_width=True)
                
                # Pilihan untuk delete
                st.subheader("🗑️ Hapus Tugas")
                task_to_delete = st.selectbox(
                    "Pilih tugas yang akan dihapus:",
                    options=range(len(st.session_state.tasks)),
                    format_func=lambda x: f"{st.session_state.tasks[x].name} - {st.session_state.tasks[x].category}"
                )
                
                if st.button("🗑️ Hapus Tugas", type="secondary"):
                    deleted_task = st.session_state.tasks.pop(task_to_delete)
                    st.success(f"✅ Tugas '{deleted_task.name}' berhasil dihapus!")
                    st.rerun()
