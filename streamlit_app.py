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
                st.dataframe(df, use_container_width=True, height=300)
                
                col1, col2 = st.columns(2)
                
                # EDIT TUGAS
                with col1:
                    st.subheader("✏️ Edit Tugas")
                    task_to_edit = st.selectbox(
                        "Pilih tugas yang akan diedit:",
                        options=range(len(st.session_state.tasks)),
                        format_func=lambda x: f"{st.session_state.tasks[x].name} - {st.session_state.tasks[x].category}",
                        key="edit_select"
                    )
                    
                    if task_to_edit is not None:
                        selected_task = st.session_state.tasks[task_to_edit]
                        
                        with st.form(key="edit_form"):
                            edit_name = st.text_input("Nama Kegiatan", value=selected_task.name)
                            edit_category = st.selectbox(
                                "Kategori",
                                ["Kuliah", "Tugas", "Ujian", "Pribadi", "Organisasi", "Olahraga", "Istirahat", "Makan"],
                                index=["Kuliah", "Tugas", "Ujian", "Pribadi", "Organisasi", "Olahraga", "Istirahat", "Makan"].index(selected_task.category) if selected_task.category in ["Kuliah", "Tugas", "Ujian", "Pribadi", "Organisasi", "Olahraga", "Istirahat", "Makan"] else 0
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
                                st.session_state.tasks[task_to_edit].name = edit_name
                                st.session_state.tasks[task_to_edit].category = edit_category
                                st.session_state.tasks[task_to_edit].start_time = datetime.combine(edit_start_date, edit_start_time)
                                st.session_state.tasks[task_to_edit].duration = edit_duration
                                st.session_state.tasks[task_to_edit].deadline = datetime.combine(edit_deadline, datetime.max.time())
                                st.session_state.tasks[task_to_edit].priority = edit_priority
                                st.session_state.tasks[task_to_edit].is_fixed = edit_is_fixed
                                st.session_state.tasks[task_to_edit].end_time = st.session_state.tasks[task_to_edit].start_time + timedelta(hours=edit_duration)
                                
                                st.success(f"✅ Tugas '{edit_name}' berhasil diupdate!")
                                st.rerun()
                
                # HAPUS TUGAS
                with col2:
                    st.subheader("🗑️ Hapus Tugas")
                    task_to_delete = st.selectbox(
                        "Pilih tugas yang akan dihapus:",
                        options=range(len(st.session_state.tasks)),
                        format_func=lambda x: f"{st.session_state.tasks[x].name} - {st.session_state.tasks[x].category}",
                        key="delete_select"
                    )
                    
                    st.write("")  # spacing
                    st.write("")
                    st.write("")
                    
                    if st.button("🗑️ Hapus Tugas", type="secondary", use_container_width=True):
                        deleted_task = st.session_state.tasks.pop(task_to_delete)
                        st.success(f"✅ Tugas '{deleted_task.name}' berhasil dihapus!")
                        st.rerun()
        
        # TAB: Import CSV
        with tab3:
            st.subheader("📤 Import Tugas dari CSV")
            
            st.markdown("""
            **Format CSV yang dibutuhkan:**
            - Nama, Kategori, Tanggal Mulai (YYYY-MM-DD), Jam Mulai (HH:MM), Durasi (jam), Deadline (YYYY-MM-DD), Prioritas (1-5), Fixed (True/False)
            
            **Contoh:**
            ```
            Nama,Kategori,Tanggal Mulai,Jam Mulai,Durasi,Deadline,Prioritas,Fixed
            Kuliah Kalkulus,Kuliah,2025-01-15,08:00,2,2025-01-15,4,True
            Tugas Algoritma,Tugas,2025-01-16,14:00,3,2025-01-20,5,False
            ```
            """)
            
            uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    
                    # Validasi kolom
                    required_cols = ['Nama', 'Kategori', 'Tanggal Mulai', 'Jam Mulai', 'Durasi', 'Deadline', 'Prioritas']
                    if not all(col in df_upload.columns for col in required_cols):
                        st.error("❌ Format CSV tidak sesuai. Pastikan memiliki kolom yang benar.")
                    else:
                        st.success(f"✅ File berhasil dibaca! Ditemukan {len(df_upload)} tugas.")
                        st.dataframe(df_upload)
                        
                        if st.button("📥 Import Semua Tugas", type="primary"):
                            for _, row in df_upload.iterrows():
                                start_datetime = datetime.strptime(
                                    f"{row['Tanggal Mulai']} {row['Jam Mulai']}", 
                                    "%Y-%m-%d %H:%M"
                                )
                                deadline_datetime = datetime.strptime(row['Deadline'], "%Y-%m-%d")
                                is_fixed_val = row.get('Fixed', False)
                                if isinstance(is_fixed_val, str):
                                    is_fixed_val = is_fixed_val.lower() == 'true'
                                
                                new_task = Task(
                                    id=st.session_state.task_counter,
                                    name=row['Nama'],
                                    category=row['Kategori'],
                                    start_time=start_datetime,
                                    duration=float(row['Durasi']),
                                    deadline=deadline_datetime,
                                    priority=int(row['Prioritas']),
                                    is_fixed=is_fixed_val
                                )
                                
                                st.session_state.tasks.append(new_task)
                                st.session_state.task_counter += 1
                            
                            st.success(f"✅ Berhasil import {len(df_upload)} tugas!")
                            st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error saat membaca CSV: {str(e)}")
    
    # ===================================================================================
    # MENU 2: GENERATE JADWAL
    # ===================================================================================
    elif menu == "🤖 Generate Jadwal":
        st.header("🤖 Generate Jadwal Otomatis")
        
        if len(st.session_state.tasks) == 0:
            st.warning("⚠️ Belum ada tugas untuk dijadwalkan. Silakan tambah tugas terlebih dahulu.")
        else:
            st.info(f"📊 Total tugas yang akan dijadwalkan: {len(st.session_state.tasks)}")
            
            # Pilih algoritma
            st.subheader("🎯 Pilih Algoritma Penjadwalan")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                algorithm = st.selectbox(
                    "Algoritma:",
                    [
                        "Genetic Algorithm (GA) - RECOMMENDED ⭐",
                        "Greedy - Earliest Deadline First (EDF)",
                        "Dynamic Programming (DP)",
                        "Brute Force (untuk ≤ 10 tugas)"
                    ]
                )
                
                # Info algoritma
                if "Genetic" in algorithm:
                    st.markdown("""
                    **🧬 Genetic Algorithm:**
                    - ✅ Paling optimal untuk dataset besar
                    - ✅ Dapat handle constraint kompleks
                    - ✅ Multi-objektif optimization
                    - ⚡ Kompleksitas: O(g × p × n)
                    - 🎯 Cocok untuk: Semua ukuran dataset
                    """)
                elif "Greedy" in algorithm:
                    st.markdown("""
                    **⚡ Greedy EDF:**
                    - ✅ Cepat dan efisien
                    - ✅ Hasil cukup optimal
                    - ✅ Intuitif (deadline terdekat dulu)
                    - ⚡ Kompleksitas: O(n log n)
                    - 🎯 Cocok untuk: Dataset sedang-besar
                    """)
                elif "Dynamic" in algorithm:
                    st.markdown("""
                    **🎲 Dynamic Programming:**
                    - ✅ Optimal secara matematis
                    - ✅ Weighted interval scheduling
                    - ⚠️ Lebih lambat dari Greedy
                    - ⚡ Kompleksitas: O(n²)
                    - 🎯 Cocok untuk: Optimasi maksimum value
                    """)
                else:  # Brute Force
                    st.markdown("""
                    **🔨 Brute Force:**
                    - ✅ Mencoba semua kemungkinan
                    - ✅ Hasil optimal (jika selesai)
                    - ⚠️ SANGAT LAMBAT untuk > 10 tugas
                    - ⚡ Kompleksitas: O(n!)
                    - 🎯 Cocok untuk: Dataset sangat kecil
                    """)
            
            with col2:
                start_date = st.date_input("Mulai dari:", datetime.now())
                days_range = st.number_input("Rentang (hari):", 1, 90, 14)
                end_date = start_date + timedelta(days=days_range)
            
            # Tombol generate
            if st.button("🚀 Generate Jadwal Optimal", type="primary", use_container_width=True):
                with st.spinner("⏳ Sedang mengoptimalkan jadwal... Mohon tunggu."):
                    start_datetime = datetime.combine(start_date, datetime.min.time())
                    end_datetime = datetime.combine(end_date, datetime.max.time())
                    
                    # Pilih algoritma
                    if "Genetic" in algorithm:
                        scheduled = genetic_algorithm_scheduling(
                            st.session_state.tasks, start_datetime, end_datetime
                        )
                        st.session_state.algorithm_used = "Genetic Algorithm"
                    elif "Greedy" in algorithm:
                        scheduled = greedy_edf_scheduling(
                            st.session_state.tasks, start_datetime, end_datetime
                        )
                        st.session_state.algorithm_used = "Greedy EDF"
                    elif "Dynamic" in algorithm:
                        scheduled = dynamic_programming_scheduling(
                            st.session_state.tasks, start_datetime, end_datetime
                        )
                        st.session_state.algorithm_used = "Dynamic Programming"
                    else:  # Brute Force
                        scheduled = brute_force_scheduling(
                            st.session_state.tasks, start_datetime, end_datetime
                        )
                        st.session_state.algorithm_used = "Brute Force"
                    
                    st.session_state.scheduled_tasks = scheduled
                    
                    # Hitung statistik
                    stats = calculate_statistics(scheduled, start_datetime, end_datetime)
                    
                    st.success("✅ Jadwal berhasil di-generate!")
                    
                    # Tampilkan statistik
                    st.subheader("📊 Statistik Jadwal")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Tugas", stats['total_tasks'])
                    col2.metric("Total Jam", f"{stats['total_hours']:.1f}")
                    col3.metric("Jam Kosong", f"{stats['free_hours']:.1f}")
                    col4.metric("Konflik", stats['conflicts'], 
                               delta=None if stats['conflicts'] == 0 else "⚠️")
                    
                    col5, col6 = st.columns(2)
                    col5.metric("✅ Tepat Waktu", stats['on_time_tasks'])
                    col6.metric("⚠️ Terlambat", stats['late_tasks'],
                               delta=None if stats['late_tasks'] == 0 else "!")
                    
                    if stats['conflicts'] > 0:
                        st.warning(f"⚠️ Ditemukan {stats['conflicts']} konflik waktu. Pertimbangkan untuk menyesuaikan durasi atau deadline.")
    
    # ===================================================================================
    # MENU 3: LIHAT JADWAL
    # ===================================================================================
    elif menu == "📊 Lihat Jadwal":
        st.header("📊 Visualisasi Jadwal")
        
        if len(st.session_state.scheduled_tasks) == 0:
            st.warning("⚠️ Belum ada jadwal yang di-generate. Silakan generate jadwal terlebih dahulu.")
        else:
            st.info(f"🤖 Algoritma yang digunakan: **{st.session_state.algorithm_used}**")
            
            # Pilih tampilan
            view_type = st.radio(
                "Pilih Tampilan:",
                ["📅 Per Minggu", "📆 Per Bulan"],
                horizontal=True
            )
            
            if view_type == "📅 Per Minggu":
                st.subheader("Jadwal Mingguan")
                
                # Group by week
                tasks_by_week = {}
                for task in st.session_state.scheduled_tasks:
                    week_start = task.start_time - timedelta(days=task.start_time.weekday())
                    week_key = week_start.strftime("%Y-%m-%d")
                    
                    if week_key not in tasks_by_week:
                        tasks_by_week[week_key] = []
                    tasks_by_week[week_key].append(task)
                
                # Pilih minggu
                selected_week = st.selectbox(
                    "Pilih Minggu:",
                    options=sorted(tasks_by_week.keys()),
                    format_func=lambda x: f"Minggu {datetime.strptime(x, '%Y-%m-%d').strftime('%d %B %Y')}"
                )
                
                # Tampilkan jadwal dalam format tabel horizontal (Senin - Minggu)
                week_tasks = tasks_by_week[selected_week]
                week_start = datetime.strptime(selected_week, "%Y-%m-%d")
                
                # Buat 7 kolom untuk 7 hari
                days_name = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
                cols = st.columns(7)
                
                for day_offset in range(7):
                    current_day = week_start + timedelta(days=day_offset)
                    day_tasks = [t for t in week_tasks if t.start_time.date() == current_day.date()]
                    
                    with cols[day_offset]:
                        # Header hari
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #FF8C00 0%, #FF6347 100%); 
                                    color: white; padding: 8px; border-radius: 8px 8px 0 0; 
                                    text-align: center; font-weight: bold; margin-bottom: 0;">
                            {days_name[day_offset]}<br>
                            <span style="font-size: 0.9em;">{current_day.strftime('%d/%m')}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Container untuk tasks
                        if len(day_tasks) == 0:
                            st.markdown("""
                            <div style="background-color: #f8f9fa; padding: 15px; 
                                        border: 1px solid #dee2e6; border-top: none;
                                        border-radius: 0 0 8px 8px; min-height: 200px;
                                        text-align: center; color: #999;">
                                <br><br>📭<br>Tidak ada kegiatan
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            tasks_html = '<div style="background-color: white; padding: 8px; border: 1px solid #dee2e6; border-top: none; border-radius: 0 0 8px 8px; min-height: 200px;">'
                            
                            for task in sorted(day_tasks, key=lambda x: x.start_time):
                                color = get_category_color(task.category)
                                emoji = get_category_emoji(task.category)
                                
                                tasks_html += f"""
                                <div style="background-color: {color}; color: white; 
                                            padding: 6px; border-radius: 5px; margin: 5px 0;
                                            font-size: 0.85em;">
                                    <b>{emoji} {task.name[:20]}</b><br>
                                    ⏰ {task.start_time.strftime('%H:%M')}-{task.end_time.strftime('%H:%M')}<br>
                                    {'⭐' * task.priority}
                                </div>
                                """
                            
                            tasks_html += '</div>'
                            st.markdown(tasks_html, unsafe_allow_html=True)
            
            else:  # Per Bulan
                st.subheader("Kalender Bulanan")
                
                # Group by month
                tasks_by_month = {}
                for task in st.session_state.scheduled_tasks:
                    month_key = task.start_time.strftime("%Y-%m")
                    
                    if month_key not in tasks_by_month:
                        tasks_by_month[month_key] = []
                    tasks_by_month[month_key].append(task)
                
                # Pilih bulan
                selected_month = st.selectbox(
                    "Pilih Bulan:",
                    options=sorted(tasks_by_month.keys()),
                    format_func=lambda x: datetime.strptime(x, "%Y-%m").strftime("%B %Y")
                )
                
                # Parse bulan
                year, month = map(int, selected_month.split('-'))
                month_tasks = tasks_by_month[selected_month]
                
                # Buat kalender
                cal = calendar.monthcalendar(year, month)
                
                st.markdown(f"""
                <div style="text-align: center; background: linear-gradient(135deg, #FF8C00 0%, #FF6347 100%);
                            color: white; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                    <h2 style="margin: 0;">{calendar.month_name[month]} {year}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # CSS untuk kalender yang lebih rapi dan efisien
                st.markdown("""
                <style>
                    .cal-header {
                        background: #FF8C00;
                        color: white;
                        padding: 10px;
                        text-align: center;
                        font-weight: bold;
                        border: 1px solid #ddd;
                        font-size: 0.9em;
                    }
                    .cal-day {
                        background: white;
                        border: 1px solid #ddd;
                        padding: 8px;
                        min-height: 100px;
                        max-height: 120px;
                        overflow-y: auto;
                        font-size: 0.8em;
                    }
                    .cal-day-empty {
                        background: #f5f5f5;
                        border: 1px solid #ddd;
                        min-height: 100px;
                    }
                    .cal-day-number {
                        font-weight: bold;
                        font-size: 1.1em;
                        color: #333;
                        margin-bottom: 5px;
                    }
                    .cal-task {
                        background: #FFE4B5;
                        padding: 3px 5px;
                        border-radius: 3px;
                        margin: 2px 0;
                        font-size: 0.75em;
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                    }
                    .cal-task-more {
                        color: #FF6347;
                        font-weight: bold;
                        font-size: 0.7em;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                # Header hari dalam HTML table
                days = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
                
                calendar_html = '<table style="width: 100%; border-collapse: collapse; table-layout: fixed;">'
                
                # Header row
                calendar_html += '<tr>'
                for day in days:
                    calendar_html += f'<td class="cal-header">{day}</td>'
                calendar_html += '</tr>'
                
                # Calendar weeks
                for week in cal:
                    calendar_html += '<tr>'
                    for day in week:
                        if day == 0:
                            calendar_html += '<td class="cal-day-empty"></td>'
                        else:
                            current_date = datetime(year, month, day).date()
                            day_tasks = [t for t in month_tasks if t.start_time.date() == current_date]
                            
                            calendar_html += '<td class="cal-day">'
                            calendar_html += f'<div class="cal-day-number">{day}</div>'
                            
                            # Tampilkan max 3 tasks
                            for i, task in enumerate(day_tasks[:3]):
                                emoji = get_category_emoji(task.category)
                                short_name = task.name[:12] + '...' if len(task.name) > 12 else task.name
                                calendar_html += f'<div class="cal-task">{emoji} {short_name}</div>'
                            
                            if len(day_tasks) > 3:
                                calendar_html += f'<div class="cal-task-more">+{len(day_tasks)-3} lagi</div>'
                            
                            calendar_html += '</td>'
                    
                    calendar_html += '</tr>'
                
                calendar_html += '</table>'
                
                st.markdown(calendar_html, unsafe_allow_html=True)
                
                # Legend kategori
                st.markdown("---")
                st.markdown("### 🎨 Legend Kategori:")
                
                categories = list(set([t.category for t in month_tasks]))
                legend_cols = st.columns(min(4, len(categories)))
                
                for i, cat in enumerate(categories):
                    with legend_cols[i % 4]:
                        color = get_category_color(cat)
                        emoji = get_category_emoji(cat)
                        st.markdown(f"""
                        <div style="display: inline-block; background-color: {color}; 
                                    color: white; padding: 5px 12px; border-radius: 5px;
                                    margin: 3px; font-size: 0.85em;">
                            {emoji} {cat}
                        </div>
                        """, unsafe_allow_html=True)
    
    # ===================================================================================
    # MENU 4: EXPORT & NOTIFIKASI
    # ===================================================================================
    elif menu == "📥 Export & Notifikasi":
        st.header("📥 Export Jadwal & Notifikasi")
        
        if len(st.session_state.scheduled_tasks) == 0:
            st.warning("⚠️ Belum ada jadwal yang di-generate.")
        else:
            # Export options
            st.subheader("💾 Export Jadwal")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export CSV
                csv_data = export_to_csv(st.session_state.scheduled_tasks)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=f"jadwal_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Info untuk PDF
                st.info("📑 Export PDF: Gunakan Print to PDF dari browser Anda pada halaman 'Lihat Jadwal'")
            
            st.markdown("---")
            
            # Notifikasi deadline
            st.subheader("🔔 Notifikasi Deadline Mendekat")
            
            days_threshold = st.slider("Tampilkan deadline dalam berapa hari ke depan?", 1, 14, 3)
            
            upcoming = check_upcoming_deadlines(st.session_state.scheduled_tasks, days_threshold)
            
            if len(upcoming) == 0:
                st.success("✅ Tidak ada deadline mendekat dalam waktu dekat!")
            else:
                st.warning(f"⚠️ Ada {len(upcoming)} tugas dengan deadline mendekat!")
                
                for task in sorted(upcoming, key=lambda x: x.deadline):
                    days_left = (task.deadline - datetime.now()).days
                    emoji = get_category_emoji(task.category)
                    
                    urgency = "🔴 URGENT!" if days_left <= 1 else "🟡 Segera" if days_left <= 2 else "🟢 Persiapan"
                    
                    st.markdown(f"""
                    <div class="deadline-urgent" style="padding: 15px; margin: 10px 0; border-radius: 10px;">
                        <h4>{urgency} {emoji} {task.name}</h4>
                        <p><b>Deadline:</b> {task.deadline.strftime('%d %B %Y %H:%M')}</p>
                        <p><b>Sisa waktu:</b> {days_left} hari</p>
                        <p><b>Prioritas:</b> {'⭐' * task.priority}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Kegiatan selanjutnya
            st.subheader("⏭️ Kegiatan Selanjutnya")
            
            now = datetime.now()
            next_tasks = [t for t in st.session_state.scheduled_tasks if t.start_time > now]
            next_tasks.sort(key=lambda x: x.start_time)
            
            if len(next_tasks) == 0:
                st.info("📭 Tidak ada kegiatan terjadwal selanjutnya")
            else:
                next_task = next_tasks[0]
                time_until = next_task.start_time - now
                hours_until = time_until.total_seconds() / 3600
                
                emoji = get_category_emoji(next_task.category)
                color = get_category_color(next_task.category)
                
                st.markdown(f"""
                <div style="background-color: {color}; color: white; padding: 20px; 
                            border-radius: 15px; text-align: center;">
                    <h2>{emoji} {next_task.name}</h2>
                    <h3>⏰ {next_task.start_time.strftime('%A, %d %B %Y')}</h3>
                    <h3>🕐 {next_task.start_time.strftime('%H:%M')} - {next_task.end_time.strftime('%H:%M')}</h3>
                    <p style="font-size: 18px;">
                        <b>Dimulai dalam: {int(hours_until)} jam {int((hours_until % 1) * 60)} menit</b>
                    </p>
                    <p><b>Durasi:</b> {next_task.duration} jam | <b>Prioritas:</b> {'⭐' * next_task.priority}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Tampilkan 3 kegiatan berikutnya
                if len(next_tasks) > 1:
                    st.subheader("📋 3 Kegiatan Berikutnya:")
                    for i, task in enumerate(next_tasks[1:4], 1):
                        emoji = get_category_emoji(task.category)
                        st.markdown(f"""
                        **{i}. {emoji} {task.name}**  
                        📅 {task.start_time.strftime('%d %B %Y, %H:%M')} | 
                        ⏱️ {task.duration} jam | 
                        {'⭐' * task.priority}
                        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 20px;">
        <p>📚 Aplikasi Penjadwalan Mahasiswa v1.0</p>
        <p>Dibuat dengan ❤️ menggunakan Streamlit & Color Psychology</p>
        <p><small>💡 Tips: Generate ulang jadwal dengan algoritma berbeda untuk hasil optimal!</small></p>
    </div>
    """, unsafe_allow_html=True)

# ===================================================================================
# RUN APPLICATION
# ===================================================================================
if __name__ == "__main__":
    main()
