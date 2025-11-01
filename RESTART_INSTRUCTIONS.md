# Restart Instructions - Apply New ORB Settings

## 🔄 **You Need to Restart**

The improved ORB settings (12 pyramid levels, scale 1.1) were just committed, but your old process is still running with the old settings (8 levels, scale 1.2).

**That's why you're still seeing only 695 features!**

---

## ⚡ **Quick Restart**

```bash
# 1. Stop current process
# Press Ctrl+C or 'q' in the window

# 2. Start fresh
./run_orby.sh
```

---

## ✅ **You'll Know It's Working When You See:**

### On Startup:
```
📊 ORB configured for typical environments:
   • 5000 features target
   • 12 pyramid levels (high coverage)       ← NEW!
   • Scale factor: 1.1 (fine-grained)        ← NEW!
   → Should detect 2500-4000 features in normal rooms
```

### During Operation:
```
detector: ORB , #features: 2500-3500    ← Much higher!
state: OK                                ← No more RELOCALIZE!
```

**If you still see 695 features, the old code is running!**

---

## 🎯 **Expected Improvement**

| Metric | Old Settings | New Settings |
|--------|-------------|--------------|
| Pyramid levels | 8 | **12** |
| Scale factor | 1.2 | **1.1** |
| Features in normal room | 695 ❌ | **2500-3500** ✅ |

---

## 🚀 **After Restart:**

You should immediately see:
- 3-4x more features detected
- "state: OK" instead of "RELOCALIZE"
- Smooth tracking in normal environments

---

**Just restart the app to get the new settings!** 🎉
