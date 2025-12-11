"""
===================================================================================
APLIKASI PENJADWALAN MAHASISWA OTOMATIS - FIXED VERSION
===================================================================================
Aplikasi ini membantu mahasiswa mengatur jadwal kegiatan dengan berbagai algoritma
optimasi. Mendukung CRUD, visualisasi kalender, dan export jadwal.

Author: Assistant
Date: 2025
Fixed: Input jam yang lebih fleksibel, tampilan kalender yang benar
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
    """Class untuk merepresentasikan satu tugas/kegiatan mahasiswa."""
    def __init__(self, id, name, category, start_time, duration, deadline, priority=3, is_fixed=False):
        self.id = id
        self.name = name
        self.category = category
        self.start_time = start_time
        self.duration = duration
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
    """Inisialisasi session state untuk menyimpan data persistent"""
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
    """Algoritma Brute Force: mencoba semua kemungkinan kombinasi jadwal."""
    if len(tasks) == 0:
        return []
    
    fixed_tasks = [t for t in tasks if t.is_fixed]
    flexible_tasks = [t for t in tasks if not t.is_fixed]
    
    if len(flexible_tasks) > 10:
        st.warning("⚠️ Brute Force tidak efisien untuk > 10 tugas flexible. Menggunakan sample.")
        flexible_tasks = flexible_tasks[:10]
    
    best_schedule = None
    best_score = float('inf')
    
    for perm in itertools.permutations(flexible_tasks):
        current_time = start_date
        scheduled = fixed_tasks.copy()
        penalty = 0
        
        for task in perm:
            while any(current_time < t.end_time and current_time + timedelta(hours=task.duration) > t.start_time 
                     for t in fixed_tasks):
                current_time += timedelta(hours=1)
            
            new_task = Task(
                task.id, task.name, task.category,
                current_time, task.duration, task.deadline,
                task.priority, task.is_fixed
            )
            
            if new_task.end_time > task.deadline:
                penalty += (new_task.end_time - task.deadline).days * task.priority * 10
            
            scheduled.append(new_task)
            current_time = new_task.end_time
        
        if penalty < best_score:
            best_score = penalty
            best_schedule = scheduled
    
    return best_schedule if best_schedule else fixed_tasks

# ===================================================================================
# ALGORITMA 2: GREEDY EARLIEST DEADLINE FIRST (EDF)
# ===================================================================================
def greedy_edf_scheduling(tasks: List[Task], start_date: datetime, end_date: datetime) -> List[Task]:
    """Algoritma Greedy - Earliest Deadline First (EDF)"""
    if len(tasks) == 0:
        return []
    
    fixed_tasks = [t for t in tasks if t.is_fixed]
    flexible_tasks = [t for t in tasks if not t.is_fixed]
    
    flexible_tasks.sort(key=lambda x: (x.deadline, -x.priority))
    
    scheduled = fixed_tasks.copy()
    current_time = start_date
    
    for task in flexible_tasks:
        while True:
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
    """Algoritma Dynamic Programming untuk Weighted Interval Scheduling"""
    if len(tasks) == 0:
        return []
    
    fixed_tasks = [t for t in tasks if t.is_fixed]
    flexible_tasks = [t for t in tasks if not t.is_fixed]
    
    if len(flexible_tasks) == 0:
        return fixed_tasks
    
    current_time = start_date
    for task in flexible_tasks:
        task.start_time = current_time
        task.end_time = current_time + timedelta(hours=task.duration)
        current_time = task.end_time
    
    flexible_tasks.sort(key=lambda x: x.end_time)
    n = len(flexible_tasks)
    
    weights = []
    for task in flexible_tasks:
        delay_factor = max(0, 1 - (task.end_time - task.deadline).days / 7)
        weight = task.priority * delay_factor
        weights.append(weight)
    
    def find_latest_compatible(index):
        for j in range(index - 1, -1, -1):
            if flexible_tasks[j].end_time <= flexible_tasks[index].start_time:
                return j
        return -1
    
    dp = [0] * n
    dp[0] = weights[0]
    
    for i in range(1, n):
        include = weights[i]
        latest_compatible = find_latest_compatible(i)
        if latest_compatible != -1:
            include += dp[latest_compatible]
        
        exclude = dp[i - 1]
        dp[i] = max(include, exclude)
    
    selected = []
    i = n - 1
    while i >= 0:
        if i == 0:
            if dp[i] > 0:
                selected.append(flexible_tasks[i])
            break
        
        latest_compatible = find_latest_compatible(i)
        include = weights[i] + (dp[latest_compatible] if latest_compatible != -1 else 0)
        exclude = dp[i - 1]
        
        if include > exclude:
            selected.append(flexible_tasks[i])
            i = latest_compatible
        else:
            i -= 1
    
    selected.reverse()
    all_tasks = fixed_tasks + selected
    all_tasks.sort(key=lambda x: x.start_time)
    
    return all_tasks

# ===================================================================================
# ALGORITMA 4: GENETIC ALGORITHM (DEFAULT - PALING OPTIMAL)
# ===================================================================================
def genetic_algorithm_scheduling(tasks: List[Task], start_date: datetime, end_date: datetime, 
                                 population_size=50, generations=100) -> List[Task]:
    """Algoritma Genetika untuk Optimasi Penjadwalan"""
    if len(tasks) == 0:
        return []
    
    fixed_tasks = [t for t in tasks if t.is_fixed]
    flexible_tasks = [t for t in tasks if not t.is_fixed]
    
    if len(flexible_tasks) == 0:
        return fixed_tasks
    
    def calculate_fitness(schedule):
        fitness = 0
        
        for i, task1 in enumerate(schedule):
            for task2 in schedule[i+1:]:
                if (task1.start_time < task2.end_time and 
                    task1.end_time > task2.start_time):
                    overlap = min(task1.end_time, task2.end_time) - max(task1.start_time, task2.start_time)
                    fitness += overlap.total_seconds() / 3600 * 100
        
        for task in schedule:
            if task.end_time > task.deadline:
                days_late = (task.end_time - task.deadline).days
                fitness += days_late * task.priority * 50
        
        for task in schedule:
            if task.end_time <= task.deadline:
                days_early = (task.deadline - task.end_time).days
                fitness -= days_early * task.priority * 2
        
        return fitness
    
    def create_individual():
        schedule = fixed_tasks.copy()
        shuffled = flexible_tasks.copy()
        random.shuffle(shuffled)
        
        for task in shuffled:
            days_range = (end_date - start_date).days
            random_days = random.randint(0, max(0, days_range - 1))
            random_hours = random.randint(8, 20)
            
            start = start_date + timedelta(days=random_days, hours=random_hours)
            new_task = Task(
                task.id, task.name, task.category,
                start, task.duration, task.deadline,
                task.priority, task.is_fixed
            )
            schedule.append(new_task)
        
        return schedule
    
    population = [create_individual() for _ in range(population_size)]
    
    for generation in range(generations):
        fitness_scores = [(schedule, calculate_fitness(schedule)) for schedule in population]
        fitness_scores.sort(key=lambda x: x[1])
        
        elite_count = max(2, population_size // 10)
        new_population = [schedule for schedule, _ in fitness_scores[:elite_count]]
        
        while len(new_population) < population_size:
            tournament_size = 5
            parent1 = min(random.sample(fitness_scores, tournament_size), key=lambda x: x[1])[0]
            parent2 = min(random.sample(fitness_scores, tournament_size), key=lambda x: x[1])[0]
            
            crossover_point = len(flexible_tasks) // 2
            child_flexible = []
            
            parent1_flexible = [t for t in parent1 if not t.is_fixed][:crossover_point]
            child_flexible.extend(parent1_flexible)
            
            parent2_flexible = [t for t in parent2 if not t.is_fixed]
            used_ids = {t.id for t in child_flexible}
            for t in parent2_flexible:
                if t.id not in used_ids:
                    child_flexible.append(t)
            
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
    
    best_schedule = min(population, key=calculate_fitness)
    best_schedule.sort(key=lambda x: x.start_time)
    
    return best_schedule

# ===================================================================================
# FUNGSI: DETEKSI KONFLIK JADWAL
# ===================================================================================
def detect_conflicts(tasks: List[Task]) -> List[Tuple[Task, Task]]:
    """Deteksi konflik waktu antar tugas"""
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
    """Hitung statistik dari jadwal yang dibuat"""
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
    """Export jadwal ke format CSV"""
    df = pd.DataFrame([t.to_dict() for t in tasks])
    return df.to_csv(index=False)

# ===================================================================================
# FUNGSI: CEK DEADLINE MENDEKAT
# ===================================================================================
def check_upcoming_deadlines(tasks: List[Task], days_threshold=3) -> List[Task]:
    """Cek tugas dengan deadline mendekat dalam N hari"""
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
    """Return warna hex berdasarkan kategori (Color Psychology)"""
    color_map = {
        'Kuliah': '#4169E1',
        'Tugas': '#FF8C00',
        'Ujian': '#DC143C',
        'Pribadi': '#32CD32',
        'Organisasi': '#9370DB',
        'Olahraga': '#20B2AA',
        'Istirahat': '#87CEEB',
        'Makan': '#FFD700',
    }
    return color_map.get(category, '#808080')

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
    
    init_session_state()
    
    st.markdown('<div class="header-style">📅 Sistem Penjadwalan Mahasiswa Otomatis</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
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
                col_time1, col_time2 = st.columns(2)
                with col_time1:
                    start_hour = st.number_input("Jam", min_value=0, max_value=23, value=8, step=1)
                with col_time2:
                    start_minute = st.number_input("Menit", min_value=0, max_value=59, value=0, step=15)
                start_time = datetime.strptime(f"{start_hour:02d}:{start_minute:02d}", "%H:%M").time()
                
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
        
        with tab2:
            st.subheader("Daftar Tugas")
            
            if len(st.session_state.tasks) == 0:
                st.info("📭 Belum ada tugas. Silakan tambah tugas terlebih dahulu.")
            else:
                tasks_data = [t.to_dict() for t in st.session_state.tasks]
                df = pd.DataFrame(tasks_data)
                st.dataframe(df, use_container_width=True, height=300)
                
                col1, col2 = st.columns(2)
                
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
                                col_time1, col_time2 = st.columns(2)
                                with col_time1:
                                    edit_hour = st.number_input("Jam", min_value=0, max_value=23, value=selected_task.start_time.hour, step=1, key="edit_hour")
                                with col_time2:
                                    edit_minute = st.number_input("Menit", min_value=0, max_value=59, value=selected_task.start_time.minute, step=15, key="edit_minute")
                            with edit_col2:
                                edit_duration = st.number_input("Durasi (jam)", min_value=0.5, max_value=24.0, value=float(selected_task.duration), step=0.5)
                                edit_deadline = st.date_input("Deadline", value=selected_task.deadline.date())
                            
                            edit_priority = st.slider("Prioritas", 1, 5, selected_task.priority)
                            edit_is_fixed = st.checkbox("Waktu Fixed", value=selected_task.is_fixed)
                            
                            submit_edit = st.form_submit_button("💾 Simpan Perubahan", type="primary")
                            
                            if submit_edit:
                                edit_start_time = datetime.strptime(f"{edit_hour:02d}:{edit_minute:02d}", "%H:%M").time()
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
                
                with col2:
                    st.subheader("🗑️ Hapus Tugas")
                    task_to_delete = st.selectbox(
                        "Pilih tugas yang akan dihapus:",
                        options=range(len(st.session_state.tasks)),
                        format_func=lambda x: f"{st.session_state.tasks[x].name} - {st.session_state.tasks[x].category}",
                        key="delete_select"
                    )
                    
                    st.write("")
                    st.write("")
                    st.write("")
                    
                    if st.button("🗑️ Hapus Tugas", type="secondary", use_container_width=True):
                        deleted_task = st.session_state.tasks.pop(task_to_delete)
                        st.success(f"✅ Tugas '{deleted_task.name}' berhasil dihapus!")
                        st.rerun()
        
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
                                deadline_datetime = datetime.strptime(row['Deadline'], "%Y-%m
