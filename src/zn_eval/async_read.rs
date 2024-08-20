use std::cell::{RefCell, Cell};
use std::collections::VecDeque;
use std::ffi::{c_void, OsString};
use std::os::windows::prelude::OsStrExt;
use std::sync::atomic::AtomicU64;
use std::time::Instant;
use windows_sys::Win32::Foundation::{HANDLE, GetLastError, GENERIC_READ, WAIT_OBJECT_0, CloseHandle, INVALID_HANDLE_VALUE, WAIT_FAILED, WIN32_ERROR};
use windows_sys::Win32::System::IO::{OVERLAPPED, OVERLAPPED_0, OVERLAPPED_0_0};
use windows_sys::Win32::Storage::FileSystem::{ReadFileEx, CreateFileW, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, FILE_FLAG_NO_BUFFERING};
use windows_sys::Win32::System::Threading::{SetEvent, CreateEventW, WaitForSingleObjectEx, INFINITE};
use append_only_vec::AppendOnlyVec;

extern crate windows_sys;

const SECTOR_SIZE: usize = 4096;
const READTASK_COUNT: usize = 48;

pub struct ReadPosition<Index> {
    pub read_request_index: Index,
    pub index_in_file: usize,
    pub byte_length: usize
}

///
/// A task to read from a queue of read requests from a single file.
/// 
pub struct ReadTask<'context, Index, F>
    where F: for<'a> FnMut(&'a [u8], ReadPosition<Index>)
{
    buffer: Box<[u8]>,
    callback: &'context RefCell<F>,
    current: Option<ReadPosition<Index>>,
    event: HANDLE,
    file: HANDLE,
    overlapped: OVERLAPPED,
    // number of currently unfinished tasks
    final_callbacks_counter: &'context Cell<usize>,
    indices: &'context RefCell<VecDeque<ReadPosition<Index>>>
}

impl<'context, Index, F> ReadTask<'context, Index, F>
    where F: for<'a> FnMut(&'a [u8], ReadPosition<Index>)
{
    fn new(event: HANDLE, file: HANDLE, indices: &'context RefCell<VecDeque<ReadPosition<Index>>>, callback: &'context RefCell<F>, final_callbacks_counter: &'context Cell<usize>) -> Self {
        Self {
            // if the range-to-read is not within a single sector, we have to read two sectors, so allocate space for two sectors
            buffer: (0..(2 * SECTOR_SIZE)).map(|_| 0).collect::<Vec<_>>().into_boxed_slice(),
            callback,
            current: None,
            event: event,
            file: file,
            overlapped: OVERLAPPED { Internal: 0, InternalHigh: 0, Anonymous: OVERLAPPED_0 { Anonymous: OVERLAPPED_0_0 { Offset: 0, OffsetHigh: 0 } }, hEvent: 0 },
            indices: indices,
            final_callbacks_counter: final_callbacks_counter
        }
    }

    unsafe fn callback(&mut self) {
        let current = self.current.take().unwrap();
        let begin = current.index_in_file % SECTOR_SIZE;
        let end = begin + current.byte_length;
        {
            (*self.callback).borrow_mut()(&self.buffer[begin..end], current);
        }
        let next_indices = {
            (*self.indices).borrow_mut().pop_front()
        };
        if let Some(to_read_next) = next_indices {
            self.run(to_read_next);
        } else {
            if (*self.final_callbacks_counter).update(|x| x - 1) == 0 {
                let result = SetEvent(self.event);
                if result == 0 {
                    panic!("SetEvent failed (error code {})", GetLastError());
                }
            }
        }
    }

    unsafe fn run(&mut self, to_read: ReadPosition<Index>) {
    
        let bytes_to_read = to_read.byte_length;
        let byte_index = to_read.index_in_file;
        let sector_index = byte_index / SECTOR_SIZE;
        let sector_byte_index = sector_index * SECTOR_SIZE;

        self.current = Some(to_read);

        debug_assert!((byte_index % SECTOR_SIZE) + bytes_to_read < 2 * SECTOR_SIZE);
        let sectors_to_read = if (byte_index % SECTOR_SIZE) + bytes_to_read >= SECTOR_SIZE { 2 } else { 1 };
        READ_SECTORS.fetch_add(sectors_to_read, std::sync::atomic::Ordering::Relaxed);

        self.overlapped.Anonymous.Anonymous.Offset = (sector_byte_index % (1 << 32)) as u32;
        self.overlapped.Anonymous.Anonymous.OffsetHigh = (sector_byte_index / (1 << 32)) as u32;
        self.overlapped.hEvent = self as *mut _ as isize;

        let result = ReadFileEx(
            self.file, 
            self.buffer.as_mut_ptr() as *mut c_void, 
            sectors_to_read as u32 * SECTOR_SIZE as u32, 
            &mut self.overlapped as *mut OVERLAPPED, 
            Some(callback_routine::<Index, F>)
        );
        if result == 0 {
            panic!("ReadFileEx failed (error code {})", GetLastError());
        }
    }
}

unsafe extern "system" fn callback_routine<Index, F>(dwerrorcode: u32, _dwnumberofbytestransfered: u32, lpoverlapped: *mut OVERLAPPED)
    where F: for<'a> FnMut(&'a [u8], ReadPosition<Index>)
{
    if dwerrorcode != 0 {
        panic!("ReadFileEx callback called with error (error code {})", dwerrorcode);
    }
    let task_ptr = lpoverlapped.as_ref().unwrap().hEvent;
    (task_ptr as *mut ReadTask<Index, F>).as_mut().unwrap().callback();
}

type CallbackType<'env, Index> = Box<dyn 'env + for<'a> FnMut(&'a [u8], ReadPosition<Index>)>;

///
/// Manages multiple queues of read requests to disk that are performed asynchronously by the OS.
/// 
pub struct DiskReadContext<'env, 'context, 'tasks, Index = usize> {
    // all the boxes are necessary to ensure that pointers stay constant
    files: &'context AppendOnlyVec<(isize, isize, Box<RefCell<VecDeque<ReadPosition<Index>>>>, RefCell<CallbackType<'env, Index>>, Box<Cell<usize>>)>,
    read_tasks: &'tasks RefCell<Vec<Vec<Box<ReadTask<'context, Index, CallbackType<'env, Index>>>>>>
}

#[derive(Clone, Copy)]
pub struct FileReader<'env, 'context, 'tasks, 'parent, Index> {
    parent: &'parent DiskReadContext<'env, 'context, 'tasks, Index>,
    parent_file_index: usize
}

impl<'env, 'context, 'tasks, 'parent, Index> FileReader<'env, 'context, 'tasks, 'parent, Index> {

    pub fn submit(&mut self, read_identifier: Index, read_at: usize, bytes_to_read: usize) {
        assert!(bytes_to_read <= 4096);
        self.parent.submit_to_file(self.parent_file_index, ReadPosition { read_request_index: read_identifier, index_in_file: read_at, byte_length: bytes_to_read });
    }
}

impl<'env, 'context, 'tasks, Index> DiskReadContext<'env, 'context, 'tasks, Index> {

    unsafe fn new(
        files: &'context AppendOnlyVec<(isize, isize, Box<RefCell<VecDeque<ReadPosition<Index>>>>, RefCell<CallbackType<'env, Index>>, Box<Cell<usize>>)>,
        tasks: &'tasks RefCell<Vec<Vec<Box<ReadTask<'context, Index, CallbackType<'env, Index>>>>>>
    ) -> Self {
        DiskReadContext { files: files, read_tasks: tasks }
    }

    pub fn open_file<'parent, F>(&'parent self, filename: &str, callback: F) -> FileReader<'env, 'context, 'tasks, 'parent, Index>
        where F: 'env + for<'a> FnMut(&'a [u8], ReadPosition<Index>)
    {
        unsafe {
            let filename_wide = OsString::from(filename).encode_wide().chain(Some(0)).collect::<Vec<_>>();
            let file = CreateFileW(
                filename_wide.as_ptr() as *const u16, 
                GENERIC_READ, 
                FILE_SHARE_READ, 
                std::ptr::null(), 
                OPEN_EXISTING, 
                FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING, 
                std::ptr::null::<()>() as isize
            );
            if file == INVALID_HANDLE_VALUE {
                panic!("CreateFileW failed (error code {})", GetLastError());
            }
            let finish_event = CreateEventW(std::ptr::null(), 0, 0, std::ptr::null());
            if finish_event == INVALID_HANDLE_VALUE {
                panic!("CreateEventW failed (error code {})", GetLastError());
            }
            let indices = Box::new(RefCell::new(VecDeque::new()));
            let callback = RefCell::new(Box::new(callback) as CallbackType<'env, Index>);
            let final_callback_counter = Box::new(Cell::new(0));

            self.files.push((file, finish_event, indices, callback, final_callback_counter));
            self.read_tasks.borrow_mut().push(Vec::new());

            let file_index = self.files.len() - 1;
            return FileReader { parent: self, parent_file_index: file_index }
        }
    }

    fn submit_to_file(&self, file_index: usize, to_read: ReadPosition<Index>) {
        let mut read_tasks = self.read_tasks.borrow_mut();
        assert!(read_tasks.len() > file_index);
        assert!(self.files.len() > file_index);
        if read_tasks[file_index].len() < READTASK_COUNT {
            // not yet maximum of readtasks reached - create a new one
            let indices_ptr: &'context RefCell<VecDeque<ReadPosition<Index>>> = &*self.files[file_index].2;
            let callback_ptr: &'context RefCell<CallbackType<'env, Index>> = &self.files[file_index].3;
            let final_callback_counter: &'context Cell<usize> = &*self.files[file_index].4;
            final_callback_counter.update(|x| x + 1);
            let read_task = ReadTask::new(self.files[file_index].1, self.files[file_index].0, indices_ptr, callback_ptr, final_callback_counter);
            read_tasks[file_index].push(Box::new(read_task));
            unsafe {
                read_tasks[file_index].last_mut().unwrap().run(to_read);
            }
        } else {
            self.files[file_index].2.borrow_mut().push_back(to_read);
        }
    }

    unsafe fn wait_for_processing_finished(&self, file_index: usize) -> Result<(), WIN32_ERROR> {
        let mut read_tasks = self.read_tasks.borrow_mut();
        assert!(read_tasks.len() > file_index);
        assert!(self.files.len() > file_index);
        let finish_event = self.files[file_index].1;

        if read_tasks[file_index].len() == 0 {
            return Ok(());
        }

        let mut wait_state = WaitForSingleObjectEx(finish_event, INFINITE, 1);
        if wait_state == WAIT_FAILED {
            return Err(GetLastError());
        }
        while wait_state != WAIT_OBJECT_0 {
            wait_state = WaitForSingleObjectEx(finish_event, INFINITE, 1);
            if wait_state == WAIT_FAILED {
                return Err(GetLastError());
            }
        }
        // ensure that all tasks are finished
        assert_eq!(self.files[file_index].4.get(), 0);
        read_tasks[file_index].clear();
        return Ok(());
    }
}

static READ_SECTORS: AtomicU64 = AtomicU64::new(0);
static READ_TIME: AtomicU64 = AtomicU64::new(0);

pub fn print_disk_read_stats() {
    println!("Read {} sectors in {} us within each thread", READ_SECTORS.load(std::sync::atomic::Ordering::SeqCst), READ_TIME.load(std::sync::atomic::Ordering::SeqCst));
    println!("Average read speed per thread: {} IOPSms", (1000. * READ_SECTORS.load(std::sync::atomic::Ordering::SeqCst) as f64 / READ_TIME.load(std::sync::atomic::Ordering::SeqCst) as f64) as i64);
}

///
/// Opens a session in which many read request to disk can be submitted asynchronously.
/// 
/// In theory, this should perform all reads started during `f()` asynchronous in the background,
/// and wait for them to complete before `perform_reads_async` terminates.
/// Currently, (almost) all of them are queued until the end of `f()`, and then performed in parallel.
/// As long as `f()` does not do much computations, this should be equivalent (and we avoid multithreading
/// for now).
/// 
/// Note that we have to wait and close resources after the reading was done, hence we implement this
/// as a function accepting a lambda. The alternative would be RAII, but in Rust RAII is not so simple.
/// 
pub fn perform_reads_async<'env, F, T, Index>(f: F) -> T
    where F: for<'a, 'b> FnOnce(&mut DiskReadContext<'env, 'a, 'b, Index>) -> T
{
    // use inner function to more easily debug lifetime mismatches
    unsafe fn internal<'env, 'context, 'tasks, F, T, Index>(
        files: &'context AppendOnlyVec<(isize, isize, Box<RefCell<VecDeque<ReadPosition<Index>>>>, RefCell<CallbackType<'env, Index>>, Box<Cell<usize>>)>,
        tasks: &'tasks RefCell<Vec<Vec<Box<ReadTask<'context, Index, CallbackType<'env, Index>>>>>>,
        f: F
    ) -> T
        where F: FnOnce(&mut DiskReadContext<'env, 'context, 'tasks, Index>) -> T
    {
        let mut readbatch = DiskReadContext::new(files, tasks);

        let result = f(&mut readbatch);

        let len = readbatch.read_tasks.borrow().len();
        for i in 0..len {
            readbatch.wait_for_processing_finished(i).unwrap();
        }
        return result;
    }
    
    unsafe {
        let files = AppendOnlyVec::new();
        // Cell is fine, since we currently do not do multithreading.
        // In particular, it is always the case that the OS callbacks are done in the same thread
        // that 
        let tasks = RefCell::new(Vec::new());

        let start = Instant::now();
        let result = internal(&files, &tasks, f);
        let end = Instant::now();
        READ_TIME.fetch_add((end - start).as_micros() as u64, std::sync::atomic::Ordering::Relaxed);

        assert_eq!(files.len(), tasks.borrow().len());
        assert!(tasks.borrow().iter().all(|tasklist| tasklist.len() == 0));
        for (file, event, _indices, _callback, _final_callback_counter) in files.iter() {
            CloseHandle(*event);
            CloseHandle(*file);
        }
        return result;
    }
}

#[cfg(test)]
use std::panic::catch_unwind;
#[cfg(test)]
use std::mem::size_of;
#[cfg(test)]
use std::fs;
#[cfg(test)]
use std::panic::UnwindSafe;

#[cfg(test)]
fn test_with_testfile<F>(testfile: &str, testfile_len: usize, base: F)
    where F: FnOnce() + UnwindSafe
{
    fs::write(testfile, (0..testfile_len).flat_map(|x| (x as u16).to_le_bytes().into_iter()).collect::<Vec<_>>()).unwrap();
    let result = catch_unwind(base);
    fs::remove_file(testfile).unwrap();
    result.unwrap();
}

#[test]
fn test_read_many() {
    let len = 65536;
    test_with_testfile("testfile_test_read_many", len, || {
        let mut actual: Vec<u16> = (0..32).map(|_| 0).collect();
        perform_reads_async(|context: &mut DiskReadContext| {
            let mut file = context.open_file("testfile_test_read_many", |x, pos| actual[pos.read_request_index] = u16::from_le_bytes([x[0], x[1]]));
            for i in 0..32 {
                file.submit(i, ((i * i * i + 7 * i) % len) * 2, size_of::<u16>());
            }
        });
        assert_eq!(
            (0..32).map(|x| (x * x * x + 7 * x) % len).map(|x| x as u16).collect::<Vec<_>>(),
            actual
        );
    })
}

#[test]
fn test_read_many_few_reads() {
    let len = 65536;
    test_with_testfile("testfile_test_read_many_few_reads", len, || {
        let mut actual: Vec<u16> = (0..128).map(|_| 0).collect();
        perform_reads_async(|context: &mut DiskReadContext| {
            let mut file = context.open_file("testfile_test_read_many_few_reads", |x, pos| actual[pos.read_request_index] = u16::from_le_bytes([x[0], x[1]]));
            for i in 0..128 {
                file.submit(i, ((i * i * i + 7 * i) % len) * 2, size_of::<u16>());
            }
        });
        assert_eq!(
            (0..128).map(|x| (x * x * x + 7 * x) % len).map(|x| x as u16).collect::<Vec<_>>(),
            actual
        );
    })
}

#[test]
fn test_read_at_sector_boundary() {
    let len = 8192;
    type ReadType = [u16; 7];

    let mut indices = (0..1170).map(|i| i * 7).collect::<Vec<_>>();
    // mess up the order
    indices.sort_by_key(|i| i & 2);

    test_with_testfile("testfile_test_read_at_sector_boundary", len, || {
        let mut actual: Vec<ReadType> = (0..1170).map(|_| [0; 7]).collect();
        perform_reads_async(|context: &mut DiskReadContext<'_, '_, '_>| {
            let mut file = context.open_file("testfile_test_read_at_sector_boundary", |x, pos| actual[pos.read_request_index] = std::array::from_fn(|i| u16::from_le_bytes([x[i * 2], x[i * 2 + 1]])));
            for (j, i) in indices.iter().enumerate() {
                file.submit(j, *i as usize * 2, size_of::<ReadType>());
            }
        });
        assert_eq!(
            indices.iter().map(|i| std::array::from_fn(|j| i + j as u16)).collect::<Vec<_>>(),
            actual
        );
    })
}

#[test]
fn test_open_file_no_read() {
    let len = 65536;
    test_with_testfile("testfile_test_open_file_no_read", len, || {
        perform_reads_async(|context: &mut DiskReadContext| {
            context.open_file("testfile_test_open_file_no_read", |_, _| {});
        });
    });
}