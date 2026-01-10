# Fix for Schema Cache Issue (PGRST204 Error)

## Problem
You're getting this error:
```
Could not find the 'content' column of 'spareparts_embeddings' in the schema cache (PGRST204)
```

This happens when:
1. The table doesn't have the `content` column (SQL script not run)
2. PostgREST schema cache is stale (needs refresh)
3. Table was created with different structure

## Solution

### Step 1: Verify and Run SQL Setup

1. Open Supabase Dashboard → SQL Editor
2. Run the SQL script: `supabase_spareparts_setup.sql`
3. Verify the table was created:
   ```sql
   SELECT column_name, data_type 
   FROM information_schema.columns 
   WHERE table_name = 'spareparts_embeddings';
   ```
   
   You should see:
   - `id` (bigint)
   - `content` (text) ← **This column is required**
   - `embedding` (vector)
   - `metadata` (jsonb)
   - `created_at` (timestamp)
   - `updated_at` (timestamp)

### Step 2: Refresh PostgREST Schema Cache

**This is the most important step!**

1. Go to Supabase Dashboard
2. Navigate to **Settings** → **API**
3. Scroll down to find **PostgREST** section or **Schema Cache** section
4. Click **"Reload Schema"** or **"Refresh Schema Cache"** button
5. Wait 30-60 seconds for the cache to refresh

**Alternative method (if button not visible):**
- Go to **Database** → **Extensions**
- Toggle the `vector` extension off and on (this forces schema refresh)
- Or restart your Supabase project (if you have access)

### Step 3: Verify Fix

Test the endpoint again:
```bash
curl -X POST "http://localhost:8000/spare-parts/upload" \
  -F "file=@your_file.xlsx"
```

## Code Improvements Made

The code now includes:

1. **Better error detection** - Catches PGRST204 and schema errors
2. **Automatic fallback** - Uses `store_embedding` method if batch insert fails
3. **Voyage AI rate limit handling** - Retries with exponential backoff
4. **Table verification** - Checks table structure before upload
5. **Clear error messages** - Provides actionable steps to fix issues

## Voyage AI Rate Limit Issue

If you see rate limit errors:
- **Free tier limits**: 3 RPM (requests per minute), 10K TPM (tokens per minute)
- **Solution**: Add payment method at https://dashboard.voyageai.com/
- **Temporary**: The code now retries with delays (60s, 120s, 180s)
- **Workaround**: Reduce `batch_size` to 10-20 to stay within limits

## Manual Schema Refresh (Alternative)

If the dashboard method doesn't work, you can refresh via SQL:

```sql
-- Force schema refresh
NOTIFY pgrst, 'reload schema';

-- Or use this query to check current schema
SELECT 
    schemaname,
    tablename,
    attname as column_name,
    typname as data_type
FROM pg_attribute a
JOIN pg_class c ON a.attrelid = c.oid
JOIN pg_type t ON a.atttypid = t.oid
WHERE c.relname = 'spareparts_embeddings'
AND a.attnum > 0
AND NOT a.attisdropped
ORDER BY a.attnum;
```

## Next Steps

After fixing the schema cache:
1. Test with a small file first (10-20 rows)
2. Monitor the logs for any remaining errors
3. Once successful, upload the full dataset
4. Consider adding payment method to Voyage AI for better rate limits

## Still Having Issues?

If the problem persists:
1. Check Supabase logs: Dashboard → Logs → Postgres Logs
2. Verify API key permissions (should have write access)
3. Try creating the table manually and checking column names match exactly
4. Contact Supabase support if schema cache refresh doesn't work